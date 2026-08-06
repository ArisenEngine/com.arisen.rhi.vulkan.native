#include "Queues/RHIVkQueue.h"
#include "Core/RHIVkDevice.h"
#include "Logger/Logger.h"
#include "RHI/Commands/RHICommandBuffer.h"
#include "Commands/RHIVkCommandBuffer.h"
#include "Commands/RHIVkCommandBufferPool.h"
#include "Descriptors/RHIVkDescriptorPool.h"
#include "Profiler.h"
#include "Definitions/RHIVkError.h"

ArisenEngine::RHI::RHIVkQueue::RHIVkQueue(RHIVkDevice* rhiDevice, VkDevice device, VkQueue queue, RHIQueueType type,
                                          IRHIDeferredDeletionQueue* deferredDeletionQueue,
                                          RHIResourceRegistry* resourceRegistry,
                                          std::shared_ptr<std::mutex> rawQueueMutex)
    : m_RHIDevice(rhiDevice), m_Device(device), m_Queue(queue), m_Type(type), m_DeferredDeletion(deferredDeletionQueue),
      m_ResourceRegistry(resourceRegistry),
      m_RawQueueMutex(std::move(rawQueueMutex))
{
    if (!m_RawQueueMutex)
        throw std::invalid_argument("RHIVkQueue requires shared raw-queue synchronization");
    CreateTimelineSemaphore();
}

ArisenEngine::RHI::RHIVkQueue::~RHIVkQueue() noexcept
{
    // Ensure GPU finished before destroying the queue semaphore.
    if (m_Queue != VK_NULL_HANDLE)
    {
        std::lock_guard<std::mutex> rawQueueLock(*m_RawQueueMutex);
        const VkResult result = vkQueueWaitIdle(m_Queue);
        if (result != VK_SUCCESS)
        {
            LOG_ERRORF("[RHIVkQueue::~RHIVkQueue]: vkQueueWaitIdle failed with {0} ({1}).",
                       GetVkResultName(result), static_cast<int>(result));
        }
    }
    if (!RetryPendingResourceReleases())
    {
        LOG_ERROR("[RHIVkQueue::~RHIVkQueue]: Pending resource retains remain for terminal device shutdown.");
    }
    if (!RetryPendingDescriptorPoolCommits())
    {
        LOG_ERROR("[RHIVkQueue::~RHIVkQueue]: Pending descriptor-pool uses remain for terminal device shutdown.");
    }
    if (m_TimelineSemaphoreHandle.IsValid())
    {
        m_RHIDevice->GetFactory()->ReleaseSemaphore(m_TimelineSemaphoreHandle);
        m_TimelineSemaphoreHandle = RHISemaphoreHandle::Invalid();
        m_TimelineSemaphore = VK_NULL_HANDLE;
    }
}

bool ArisenEngine::RHI::RHIVkQueue::TryPublishResourceRelease(
    const PendingResourceRelease& pending) noexcept
{
    if (!m_ResourceRegistry)
        return false;

    try
    {
        m_ResourceRegistry->UpdateTicket(pending.handle, m_Type, pending.ticket);
        return m_ResourceRegistry->Release(pending.handle);
    }
    catch (const std::exception& error)
    {
        LOG_ERRORF("[RHIVkQueue]: Resource release {0}:{1} for ticket {2} failed: {3}",
                   pending.handle.index, pending.handle.generation, pending.ticket, error.what());
    }
    catch (...)
    {
        LOG_ERRORF("[RHIVkQueue]: Resource release {0}:{1} for ticket {2} failed with an unknown error.",
                   pending.handle.index, pending.handle.generation, pending.ticket);
    }
    return false;
}

bool ArisenEngine::RHI::RHIVkQueue::RetryPendingResourceReleases() noexcept
{
    size_t retainedCount = 0;
    for (const PendingResourceRelease& pending : m_PendingResourceReleases)
    {
        if (!TryPublishResourceRelease(pending))
            m_PendingResourceReleases[retainedCount++] = pending;
    }
    m_PendingResourceReleases.resize(retainedCount);
    return retainedCount == 0;
}

bool ArisenEngine::RHI::RHIVkQueue::TryCommitDescriptorPoolUse(
    const PendingDescriptorPoolCommit& pending) noexcept
{
    if (m_RejectNextDescriptorPoolCommitForTesting.exchange(false, std::memory_order_acq_rel))
        return false;

    try
    {
        auto* poolRegistry = m_RHIDevice ? m_RHIDevice->GetDescriptorPoolPool() : nullptr;
        auto* poolItem = poolRegistry ? poolRegistry->Get(pending.poolHandle) : nullptr;
        auto* pool = poolItem ? static_cast<RHIVkDescriptorPool*>(poolItem->pool) : nullptr;
        if (!pool)
            return false;
        return pool->CommitPoolUse(pending.poolId, m_Type, pending.ticket);
    }
    catch (const std::exception& error)
    {
        LOG_ERRORF("[RHIVkQueue]: Descriptor-pool use {0}:{1}/{2} for ticket {3} failed: {4}",
                   pending.poolHandle.index, pending.poolHandle.generation, pending.poolId,
                   pending.ticket, error.what());
    }
    catch (...)
    {
        LOG_ERRORF("[RHIVkQueue]: Descriptor-pool use {0}:{1}/{2} for ticket {3} failed with an unknown error.",
                   pending.poolHandle.index, pending.poolHandle.generation, pending.poolId,
                   pending.ticket);
    }
    return false;
}

bool ArisenEngine::RHI::RHIVkQueue::RetryPendingDescriptorPoolCommits() noexcept
{
    size_t retainedCount = 0;
    for (const PendingDescriptorPoolCommit& pending : m_PendingDescriptorPoolCommits)
    {
        if (!TryCommitDescriptorPoolUse(pending))
            m_PendingDescriptorPoolCommits[retainedCount++] = pending;
    }
    m_PendingDescriptorPoolCommits.resize(retainedCount);
    return retainedCount == 0;
}

void ArisenEngine::RHI::RHIVkQueue::CreateTimelineSemaphore()
{
    m_TimelineSemaphoreHandle = m_RHIDevice->GetFactory()->CreateTimelineSemaphore(0);
    if (!m_TimelineSemaphoreHandle.IsValid())
    {
        LOG_FATAL_AND_THROW("[RHIVkQueue]: failed to create timeline semaphore from factory!");
    }

    auto* semItem = m_RHIDevice->GetSemaphorePool()->Get(m_TimelineSemaphoreHandle);
    if (semItem && semItem->semaphore != VK_NULL_HANDLE)
    {
        m_TimelineSemaphore = semItem->semaphore;
    }
    else
    {
        LOG_FATAL_AND_THROW("[RHIVkQueue]: timeline semaphore has invalid raw handle!");
    }
}

#include "Presentation/RHIVkSwapChain.h"

ArisenEngine::RHI::RHIGpuTicket ArisenEngine::RHI::RHIVkQueue::Submit(RHICommandBufferHandle handle,
                                                                      const RHISubmitDescriptor* descriptor)
{
    ARISEN_PROFILE_ZONE("RHI::VulkanQueueSubmit");
    std::lock_guard<std::mutex> lock(m_SubmitMutex);

    auto* commandBuffer = m_RHIDevice->GetCommandBuffer(handle);
    if (!commandBuffer)
    {
        ThrowInvalidHandle("vkQueueSubmit", "RHICommandBuffer", handle.index, handle.generation,
                           "Command buffer does not exist on the submitting device");
    }
    if (!commandBuffer->ReadyForSubmit())
    {
        ThrowInvalidState("vkQueueSubmit", "RHICommandBuffer",
                          (static_cast<uint64_t>(handle.generation) << 32) | handle.index,
                          "Command buffer is not ready for submission");
    }
    if (m_Queue == VK_NULL_HANDLE || m_TimelineSemaphore == VK_NULL_HANDLE)
    {
        ThrowInvalidState("vkQueueSubmit", "RHIVkQueue", reinterpret_cast<uint64_t>(this),
                          "Queue or timeline semaphore is not initialized");
    }
    const bool resourceReleasesCommitted = RetryPendingResourceReleases();
    const bool descriptorUsesCommitted = RetryPendingDescriptorPoolCommits();
    if (!resourceReleasesCommitted || !descriptorUsesCommitted)
    {
        ThrowInvalidState("vkQueueSubmit", "RHIVkQueue", reinterpret_cast<uint64_t>(this),
                          "A previous accepted submission still owns rejected resource or descriptor-pool publication");
    }

    RHIVkCommandBuffer* vkCmd = static_cast<RHIVkCommandBuffer*>(commandBuffer);
    auto* ownerPool = static_cast<RHIVkCommandBufferPool*>(vkCmd->GetOwner());
    if (!ownerPool)
    {
        ThrowInvalidState("vkQueueSubmit", "RHICommandBuffer",
                          (static_cast<uint64_t>(handle.generation) << 32) | handle.index,
                          "Command buffer owner pool is unavailable");
    }
    if (!vkCmd->IsCompiled())
    {
        vkCmd->Compile();
    }
    if (!m_ResourceRegistry)
    {
        ThrowInvalidState("vkQueueSubmit", "RHIVkQueue", reinterpret_cast<uint64_t>(this),
                          "Resource registry is unavailable");
    }
    m_PendingResourceReleases.reserve(
        m_PendingResourceReleases.size() + vkCmd->GetTrackedResourceHandles().size());
    m_PendingDescriptorPoolCommits.reserve(
        m_PendingDescriptorPoolCommits.size() + vkCmd->GetTrackedDescriptorPools().size());

    Containers::Vector<VkSemaphore> waitSems;
    Containers::Vector<VkPipelineStageFlags> waitStages;
    Containers::Vector<uint64_t> waitValues;
    Containers::Vector<VkSemaphore> signalSems;
    Containers::Vector<uint64_t> signalValues;
    const UInt32 resourceFrameIndex = commandBuffer->GetCurrentFrameIndex();
    const UInt32 swapChainFrameIndex = descriptor &&
        (descriptor->WaitSwapChain || descriptor->SignalSwapChain)
        ? descriptor->SwapChainFrameIndex
        : resourceFrameIndex;
    auto* waitSwapChain = descriptor && descriptor->WaitSwapChain
                              ? static_cast<RHIVkSwapChain*>(descriptor->WaitSwapChain)
                              : nullptr;
    auto* signalSwapChain = descriptor && descriptor->SignalSwapChain
                                ? static_cast<RHIVkSwapChain*>(descriptor->SignalSwapChain)
                                : nullptr;
    const bool sharedSwapChainReservation = waitSwapChain && waitSwapChain == signalSwapChain;
    RHIVkSwapChain::FrameSubmissionPlan sharedPlan{};
    RHIVkSwapChain::FrameSubmissionPlan waitPlan{};
    RHIVkSwapChain::FrameSubmissionPlan signalPlan{};
    bool sharedPrepared = false;
    bool waitPrepared = false;
    bool signalPrepared = false;
    RHIGpuTicket submitTicket = 0;

    const auto cancelSwapChainReservations = [&]() noexcept
    {
        if (sharedPrepared)
            waitSwapChain->CancelFrameSubmission(swapChainFrameIndex);
        else
        {
            if (signalPrepared)
                signalSwapChain->CancelFrameSubmission(swapChainFrameIndex);
            if (waitPrepared)
                waitSwapChain->CancelFrameSubmission(swapChainFrameIndex);
        }
    };

    try
    {
        if (descriptor)
        {
            auto* vkDevice = m_RHIDevice;
            if (descriptor->waitSemaphoreCount > 0 && !descriptor->pWaitSemaphores)
                ThrowInvalidParameter("vkQueueSubmit", "pWaitSemaphores");
            if (descriptor->signalSemaphoreCount > 0 && !descriptor->pSignalSemaphores)
                ThrowInvalidParameter("vkQueueSubmit", "pSignalSemaphores");

            for (UInt32 i = 0; i < descriptor->waitSemaphoreCount; ++i)
            {
                auto* semItem = vkDevice->GetSemaphorePool()->Get(descriptor->pWaitSemaphores[i]);
                if (!semItem)
                {
                    const auto invalid = descriptor->pWaitSemaphores[i];
                    ThrowInvalidHandle("vkQueueSubmit", "RHISemaphore", invalid.index, invalid.generation,
                                       "Explicit wait semaphore is stale");
                }
                waitSems.push_back(semItem->semaphore);
                waitStages.push_back(descriptor->pWaitDstStageMask
                                         ? descriptor->pWaitDstStageMask[i]
                                         : VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
                waitValues.push_back(descriptor->pWaitValues ? descriptor->pWaitValues[i] : 0);
            }

            for (UInt32 i = 0; i < descriptor->signalSemaphoreCount; ++i)
            {
                auto* semItem = vkDevice->GetSemaphorePool()->Get(descriptor->pSignalSemaphores[i]);
                if (!semItem)
                {
                    const auto invalid = descriptor->pSignalSemaphores[i];
                    ThrowInvalidHandle("vkQueueSubmit", "RHISemaphore", invalid.index, invalid.generation,
                                       "Explicit signal semaphore is stale");
                }
                signalSems.push_back(semItem->semaphore);
                signalValues.push_back(descriptor->pSignalValues ? descriptor->pSignalValues[i] : 0);
            }
        }

        if (sharedSwapChainReservation)
        {
            sharedPlan = waitSwapChain->PrepareFrameSubmission(swapChainFrameIndex, true, true);
            sharedPrepared = true;
        }
        else
        {
            if (waitSwapChain)
            {
                waitPlan = waitSwapChain->PrepareFrameSubmission(swapChainFrameIndex, true, false);
                waitPrepared = true;
            }
            if (signalSwapChain)
            {
                signalPlan = signalSwapChain->PrepareFrameSubmission(swapChainFrameIndex, false, true);
                signalPrepared = true;
            }
        }

        const auto appendSwapChainPlan = [&](const RHIVkSwapChain::FrameSubmissionPlan& plan)
        {
            if (plan.waitSemaphore.IsValid())
            {
                auto* semItem = m_RHIDevice->GetSemaphorePool()->Get(plan.waitSemaphore);
                if (!semItem || semItem->semaphore == VK_NULL_HANDLE)
                    ThrowInvalidHandle("vkQueueSubmit", "RHISemaphore", plan.waitSemaphore.index,
                                       plan.waitSemaphore.generation, "Swapchain wait semaphore is stale");
                waitSems.push_back(semItem->semaphore);
                waitStages.push_back(waitSwapChain && waitSwapChain->RequiresAcquireSemaphoreWait()
                                         ? VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
                                         : VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
                waitValues.push_back(0);
            }
            if (plan.signalSemaphore.IsValid())
            {
                auto* semItem = m_RHIDevice->GetSemaphorePool()->Get(plan.signalSemaphore);
                if (!semItem || semItem->semaphore == VK_NULL_HANDLE)
                    ThrowInvalidHandle("vkQueueSubmit", "RHISemaphore", plan.signalSemaphore.index,
                                       plan.signalSemaphore.generation, "Swapchain signal semaphore is stale");
                signalSems.push_back(semItem->semaphore);
                signalValues.push_back(0);
            }
        };

        if (sharedPrepared)
            appendSwapChainPlan(sharedPlan);
        else
        {
            if (waitPrepared) appendSwapChainPlan(waitPlan);
            if (signalPrepared) appendSwapChainPlan(signalPlan);
        }

        submitTicket = m_LatestTicket.load(std::memory_order_acquire) + 1;

        VkTimelineSemaphoreSubmitInfo timelineInfo{};
        timelineInfo.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.pNext = &timelineInfo;

        submitInfo.waitSemaphoreCount = static_cast<uint32_t>(waitSems.size());
        submitInfo.pWaitSemaphores = waitSems.data();
        submitInfo.pWaitDstStageMask = waitStages.data();

        timelineInfo.waitSemaphoreValueCount = submitInfo.waitSemaphoreCount;
        timelineInfo.pWaitSemaphoreValues = waitValues.data();

        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &vkCmd->m_VkCommandBuffer;

        signalSems.push_back(m_TimelineSemaphore);
        signalValues.push_back(submitTicket);

        submitInfo.signalSemaphoreCount = static_cast<UInt32>(signalSems.size());
        submitInfo.pSignalSemaphores = signalSems.data();

        timelineInfo.signalSemaphoreValueCount = static_cast<uint32_t>(signalValues.size());
        timelineInfo.pSignalSemaphoreValues = signalValues.data();

        vkCmd->SetLatestSubmitTicket(submitTicket);

        const VkResult injectedResult = static_cast<VkResult>(
            m_InjectedSubmitResult.exchange(VK_SUCCESS, std::memory_order_acq_rel));
        if (injectedResult != VK_SUCCESS)
        {
            ThrowVkFailure("vkQueueSubmit", injectedResult, "RHICommandBuffer",
                           reinterpret_cast<uint64_t>(m_Queue), handle.index, handle.generation,
                           "deterministic injected failure");
        }
        {
            std::lock_guard<std::mutex> rawQueueLock(*m_RawQueueMutex);
            CheckVkResult(
                vkQueueSubmit(m_Queue, 1, &submitInfo, VK_NULL_HANDLE),
                "vkQueueSubmit",
                "RHICommandBuffer",
                reinterpret_cast<uint64_t>(m_Queue),
                handle.index,
                handle.generation);
        }

        ownerPool->RecordAcceptedSubmission(m_Type, submitTicket);
        m_LatestTicket.store(submitTicket, std::memory_order_release);

        if (sharedPrepared)
            waitSwapChain->CommitFrameSubmission(swapChainFrameIndex, submitTicket);
        else
        {
            if (waitPrepared)
                waitSwapChain->CommitFrameSubmission(swapChainFrameIndex, submitTicket);
            if (signalPrepared)
                signalSwapChain->CommitFrameSubmission(swapChainFrameIndex, submitTicket);
        }
    }
    catch (...)
    {
        cancelSwapChainReservations();
        vkCmd->SetLatestSubmitTicket(0);
        throw;
    }

    // The submit is accepted. Transfer every captured retain to either the registry's
    // deferred queue or this queue's retry journal without allowing cleanup failure
    // to falsify the returned submit ticket.
    for (RHIResourceHandle trackedHandle : vkCmd->GetTrackedResourceHandles())
    {
        const PendingResourceRelease pending{trackedHandle, submitTicket};
        if (!TryPublishResourceRelease(pending))
            m_PendingResourceReleases.emplace_back(pending);
    }
    vkCmd->ClearTrackedResourceHandles();

    // Transfer descriptor-pool leases to accepted GPU tickets. A rejected commit
    // remains queue-owned and continues to block reset until an explicit retry.
    for (const auto& trackedPool : vkCmd->GetTrackedDescriptorPools())
    {
        const PendingDescriptorPoolCommit pending{
            trackedPool.poolHandle,
            trackedPool.poolId,
            submitTicket
        };
        if (!TryCommitDescriptorPoolUse(pending))
            m_PendingDescriptorPoolCommits.emplace_back(pending);
    }
    vkCmd->ClearTrackedDescriptorPools();
    vkCmd->SetLatestSubmitTicket(submitTicket);

    return submitTicket;
}

ArisenEngine::RHI::RHIGpuTicket ArisenEngine::RHI::RHIVkQueue::RetireSwapChainFrame(
    RHIVkSwapChain* swapChain,
    UInt32 frameIndex)
{
    ARISEN_PROFILE_ZONE("RHI::VulkanRetireSwapChainFrame");
    std::lock_guard<std::mutex> lock(m_SubmitMutex);

    if (!swapChain)
        ThrowInvalidParameter("vkQueueSubmit", "swapChain", "Retirement requires a valid swapchain");
    if (m_Queue == VK_NULL_HANDLE || m_TimelineSemaphore == VK_NULL_HANDLE)
    {
        ThrowInvalidState("vkQueueSubmit", "RHIVkQueue", reinterpret_cast<uint64_t>(this),
                          "Queue or timeline semaphore is not initialized");
    }
    const bool resourceReleasesCommitted = RetryPendingResourceReleases();
    const bool descriptorUsesCommitted = RetryPendingDescriptorPoolCommits();
    if (!resourceReleasesCommitted || !descriptorUsesCommitted)
    {
        ThrowInvalidState("vkQueueSubmit", "RHIVkQueue", reinterpret_cast<uint64_t>(this),
                          "A previous accepted submission still owns rejected resource or descriptor-pool publication");
    }

    const auto plan = swapChain->PrepareFrameRetirement(frameIndex);
    if (plan.terminal)
        return plan.previousTicket;

    RHIGpuTicket retirementTicket = plan.previousTicket;
    bool frameCommitted = false;
    try
    {
        if (plan.requiresQueueSubmit)
        {
            const RHIGpuTicket submitTicket = m_LatestTicket.load(std::memory_order_acquire) + 1;
            VkSemaphore waitSemaphores[1] = {plan.waitSemaphore};
            VkPipelineStageFlags waitStages[1] = {VK_PIPELINE_STAGE_ALL_COMMANDS_BIT};
            uint64_t waitValues[1] = {0};
            const uint32_t waitCount = plan.waitSemaphore != VK_NULL_HANDLE ? 1u : 0u;

            VkSemaphore signalSemaphores[2]{};
            uint64_t signalValues[2]{};
            uint32_t signalCount = 0;
            if (plan.signalSemaphore != VK_NULL_HANDLE)
            {
                signalSemaphores[signalCount] = plan.signalSemaphore;
                signalValues[signalCount] = 0;
                signalCount++;
            }
            signalSemaphores[signalCount] = m_TimelineSemaphore;
            signalValues[signalCount] = submitTicket;
            signalCount++;

            VkTimelineSemaphoreSubmitInfo timelineInfo{};
            timelineInfo.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
            timelineInfo.waitSemaphoreValueCount = waitCount;
            timelineInfo.pWaitSemaphoreValues = waitCount > 0 ? waitValues : nullptr;
            timelineInfo.signalSemaphoreValueCount = signalCount;
            timelineInfo.pSignalSemaphoreValues = signalValues;

            VkSubmitInfo submitInfo{};
            submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submitInfo.pNext = &timelineInfo;
            submitInfo.waitSemaphoreCount = waitCount;
            submitInfo.pWaitSemaphores = waitCount > 0 ? waitSemaphores : nullptr;
            submitInfo.pWaitDstStageMask = waitCount > 0 ? waitStages : nullptr;
            submitInfo.commandBufferCount = plan.commandBuffer != VK_NULL_HANDLE ? 1u : 0u;
            submitInfo.pCommandBuffers = plan.commandBuffer != VK_NULL_HANDLE
                                             ? &plan.commandBuffer
                                             : nullptr;
            submitInfo.signalSemaphoreCount = signalCount;
            submitInfo.pSignalSemaphores = signalSemaphores;

            const VkResult injectedResult = static_cast<VkResult>(
                m_InjectedSubmitResult.exchange(VK_SUCCESS, std::memory_order_acq_rel));
            if (injectedResult != VK_SUCCESS)
            {
                ThrowVkFailure(
                    "vkQueueSubmit",
                    injectedResult,
                    "RHIVkSwapChain",
                    reinterpret_cast<uint64_t>(swapChain),
                    UINT32_MAX,
                    0,
                    "deterministic injected frame-retirement failure");
            }
            {
                std::lock_guard<std::mutex> rawQueueLock(*m_RawQueueMutex);
                CheckVkResult(
                    vkQueueSubmit(m_Queue, 1, &submitInfo, VK_NULL_HANDLE),
                    "vkQueueSubmit",
                    "RHIVkSwapChain",
                    reinterpret_cast<uint64_t>(swapChain),
                    UINT32_MAX,
                    0,
                    "failed to consume swapchain frame synchronization");
            }

            m_LatestTicket.store(submitTicket, std::memory_order_release);
            retirementTicket = submitTicket;
        }

        const VkResult presentResult = swapChain->CommitFrameRetirement(
            frameIndex,
            retirementTicket,
            plan);
        frameCommitted = true;
        if (presentResult != VK_SUCCESS && !IsVkSwapChainRecreateResult(presentResult))
        {
            ThrowVkFailure("vkQueuePresentKHR", presentResult, "RHIVkSwapChain",
                           reinterpret_cast<uint64_t>(swapChain), UINT32_MAX, 0,
                           "frame retirement presentation failed");
        }

        return retirementTicket;
    }
    catch (...)
    {
        if (!frameCommitted)
            swapChain->CancelFrameRetirement(frameIndex);
        throw;
    }
}

void ArisenEngine::RHI::RHIVkQueue::Update()
{
    std::lock_guard<std::mutex> lock(m_SubmitMutex);
    UpdateLocked();
}

void ArisenEngine::RHI::RHIVkQueue::UpdateLocked()
{
    ARISEN_PROFILE_ZONE("RHI::VulkanQueueUpdate");
    const bool resourceReleasesCommitted = RetryPendingResourceReleases();
    const bool descriptorUsesCommitted = RetryPendingDescriptorPoolCommits();
    if (!resourceReleasesCommitted || !descriptorUsesCommitted)
    {
        ThrowInvalidState("RHIVkQueue::Update", "RHIVkQueue", reinterpret_cast<uint64_t>(this),
                          "Accepted-submit resource or descriptor-pool publication is still rejected");
    }
    if (m_TimelineSemaphore == VK_NULL_HANDLE)
    {
        return;
    }

    uint64_t completed = 0;
    CheckVkResult(vkGetSemaphoreCounterValue(m_Device, m_TimelineSemaphore, &completed),
                  "vkGetSemaphoreCounterValue", "RHIVkQueue", reinterpret_cast<uint64_t>(m_Queue));
    m_CompletedSubmitTicket.store(static_cast<RHIGpuTicket>(completed), std::memory_order_release);

    if (m_DeferredDeletion)
    {
        m_DeferredDeletion->Flush(m_Type, m_CompletedSubmitTicket.load(std::memory_order_acquire));
    }
}

void ArisenEngine::RHI::RHIVkQueue::WaitIdle()
{
    ARISEN_PROFILE_ZONE("RHI::VulkanQueueWaitIdle");
    std::lock_guard<std::mutex> lock(m_SubmitMutex);
    CheckVkResult(WaitIdleNoThrow(), "vkQueueWaitIdle", "RHIVkQueue",
                  reinterpret_cast<uint64_t>(m_Queue));
    const bool resourceReleasesCommitted = RetryPendingResourceReleases();
    const bool descriptorUsesCommitted = RetryPendingDescriptorPoolCommits();
    if (!resourceReleasesCommitted || !descriptorUsesCommitted)
    {
        ThrowInvalidState("vkQueueWaitIdle", "RHIVkQueue", reinterpret_cast<uint64_t>(this),
                          "Accepted-submit resource or descriptor-pool publication is still rejected");
    }
}

VkResult ArisenEngine::RHI::RHIVkQueue::WaitIdleNoThrow() noexcept
{
    const VkResult injectedResult = static_cast<VkResult>(
        m_InjectedWaitIdleResult.exchange(VK_SUCCESS, std::memory_order_acq_rel));
    if (injectedResult != VK_SUCCESS)
        return injectedResult;

    if (m_Queue == VK_NULL_HANDLE)
        return VK_SUCCESS;

    std::lock_guard<std::mutex> rawQueueLock(*m_RawQueueMutex);
    return vkQueueWaitIdle(m_Queue);
}

VkResult ArisenEngine::RHI::RHIVkQueue::PresentNoThrow(
    const VkPresentInfoKHR& presentInfo) noexcept
{
    if (m_Queue == VK_NULL_HANDLE)
        return VK_ERROR_INITIALIZATION_FAILED;

    std::lock_guard<std::mutex> rawQueueLock(*m_RawQueueMutex);
    const VkResult result = vkQueuePresentKHR(m_Queue, &presentInfo);
    if (result != VK_SUCCESS)
        return result;
    return VK_SUCCESS;
}

void ArisenEngine::RHI::RHIVkQueue::WaitForTicket(RHIGpuTicket ticket)
{
    if (ticket == 0)
        return;

    Update();
    if (GetCompletedTicket() >= ticket)
        return;

    const RHIGpuTicket latestTicket = GetLatestTicket();
    if (ticket > latestTicket)
    {
        ThrowInvalidState(
            "vkWaitSemaphores",
            "RHIVkQueue",
            reinterpret_cast<uint64_t>(m_Queue),
            "Cannot wait for a ticket that has not been accepted by this queue");
    }
    if (m_TimelineSemaphore == VK_NULL_HANDLE)
    {
        ThrowInvalidState(
            "vkWaitSemaphores",
            "RHIVkQueue",
            reinterpret_cast<uint64_t>(m_Queue),
            "Queue timeline semaphore is unavailable");
    }

    VkSemaphoreWaitInfo waitInfo{};
    waitInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
    waitInfo.semaphoreCount = 1;
    waitInfo.pSemaphores = &m_TimelineSemaphore;
    waitInfo.pValues = &ticket;

    CheckVkResult(vkWaitSemaphores(m_Device, &waitInfo, UINT64_MAX),
                  "vkWaitSemaphores", "RHIVkQueue", reinterpret_cast<uint64_t>(m_Queue));
    Update();
}

void ArisenEngine::RHI::RHIVkQueue::WaitForTicketUnderSubmitLock(RHIGpuTicket ticket)
{
    WaitForTicketLocked(ticket);
}

void ArisenEngine::RHI::RHIVkQueue::WaitForTicketLocked(RHIGpuTicket ticket)
{
    ARISEN_PROFILE_ZONE("RHI::VulkanQueueWait");
    if (ticket == 0)
    {
        return;
    }

    UpdateLocked();

    if (GetCompletedTicket() >= ticket)
        return;

    if (ticket > GetLatestTicket())
    {
        ThrowInvalidState(
            "vkWaitSemaphores",
            "RHIVkQueue",
            reinterpret_cast<uint64_t>(m_Queue),
            "Cannot wait for a ticket that has not been accepted by this queue");
    }

    if (m_TimelineSemaphore == VK_NULL_HANDLE)
    {
        ThrowInvalidState(
            "vkWaitSemaphores",
            "RHIVkQueue",
            reinterpret_cast<uint64_t>(m_Queue),
            "Queue timeline semaphore is unavailable");
    }

    VkSemaphoreWaitInfo waitInfo{};
    waitInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
    waitInfo.semaphoreCount = 1;
    waitInfo.pSemaphores = &m_TimelineSemaphore;
    waitInfo.pValues = &ticket;

    CheckVkResult(vkWaitSemaphores(m_Device, &waitInfo, UINT64_MAX),
                  "vkWaitSemaphores", "RHIVkQueue", reinterpret_cast<uint64_t>(m_Queue));
    UpdateLocked();
}
