using Arisen.Native.RHI;
using ArisenKernel.Contracts;

namespace ArisenEngine.RHI.Vulkan.Native;

public sealed class VulkanRHIDevice : IRHIDevice
{
    public VulkanRHIDevice(IntPtr nativeHandle)
    {
        NativeHandle = nativeHandle;
    }

    public IntPtr NativeHandle { get; }

    public bool IsValid => NativeHandle != IntPtr.Zero;

    public void WaitIdle()
    {
        if (IsValid)
        {
            RHIDeviceAPI.RHIDevice_DeviceWaitIdle(NativeHandle);
        }
    }

    public ulong SubmitCommandList(IntPtr commandBufferHandle)
    {
        return IsValid
            ? RHIDeviceAPI.RHIDevice_Submit(NativeHandle, 0, 0, commandBufferHandle)
            : 0;
    }

    public void WaitQueueTicket(ulong ticket)
    {
        if (IsValid)
        {
            RHIDeviceAPI.RHIDevice_WaitQueueTicket(NativeHandle, ticket);
        }
    }

    public ulong GetCompletedTicket()
    {
        return IsValid
            ? RHIDeviceAPI.RHIDevice_GetCompletedSubmitTicket(NativeHandle)
            : 0;
    }

    public IntPtr GetSharedWin32Handle(uint index, uint generation)
    {
        return IsValid
            ? RHIDeviceAPI.RHIDevice_GetSharedWin32Handle(NativeHandle, index, generation)
            : IntPtr.Zero;
    }
}
