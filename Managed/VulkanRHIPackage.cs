using ArisenKernel.Contracts;
using ArisenKernel.Diagnostics;
using ArisenKernel.Packages;
using ArisenKernel.Services;

namespace ArisenEngine.RHI.Vulkan.Native;

public sealed class VulkanRHIPackage : IPackageEntry
{
    public const string PackageId = "com.arisen.rhi.vulkan.native";

    private VulkanRHIBackend? m_Backend;

    public void OnLoad(IServiceRegistry services)
    {
        m_Backend = new VulkanRHIBackend();
        services.RegisterService<IRHIBackend>(m_Backend);
        KernelLog.Info("[VulkanRHIPackage] Registered Vulkan RHI backend.");
    }

    public void OnUnload(IServiceRegistry services)
    {
        m_Backend?.Shutdown();
        m_Backend = null;
        KernelLog.Info("[VulkanRHIPackage] Unloaded Vulkan RHI backend.");
    }
}
