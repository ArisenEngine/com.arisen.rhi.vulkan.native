using Arisen.Native.RHI;
using ArisenKernel.Contracts;
using System.Threading;

namespace ArisenEngine.RHI.Vulkan.Native;

public sealed class VulkanRHIDevice : IRHIDevice
{
    private IntPtr m_NativeHandle;
    private long m_Generation;

    public VulkanRHIDevice(IntPtr nativeHandle, ulong generation = 1)
    {
        Attach(nativeHandle, generation);
    }

    public IntPtr NativeHandle => ReadHandle();

    public ulong Generation => unchecked((ulong)Interlocked.Read(ref m_Generation));

    public bool IsValid => NativeHandle != IntPtr.Zero;

    internal void Attach(IntPtr nativeHandle, ulong generation)
    {
        if (nativeHandle == IntPtr.Zero)
        {
            throw new ArgumentException("A Vulkan device generation requires a valid native handle.", nameof(nativeHandle));
        }
        if (generation == 0)
        {
            throw new ArgumentOutOfRangeException(nameof(generation));
        }

        Interlocked.Exchange(ref m_Generation, unchecked((long)generation));
        Interlocked.Exchange(ref m_NativeHandle, nativeHandle);
    }

    internal void Detach()
    {
        Interlocked.Exchange(ref m_NativeHandle, IntPtr.Zero);
    }

    public void WaitIdle()
    {
        IntPtr handle = ReadHandle();
        if (handle != IntPtr.Zero)
        {
            RHIDeviceAPI.RHIDevice_DeviceWaitIdle(handle);
        }
    }

    public ulong SubmitCommandList(IntPtr commandBufferHandle)
    {
        IntPtr handle = ReadHandle();
        return handle != IntPtr.Zero
            ? RHIDeviceAPI.RHIDevice_Submit(handle, 0, 0, commandBufferHandle)
            : 0;
    }

    public void WaitQueueTicket(ulong ticket)
    {
        IntPtr handle = ReadHandle();
        if (handle != IntPtr.Zero)
        {
            RHIDeviceAPI.RHIDevice_WaitQueueTicket(handle, ticket);
        }
    }

    public ulong GetCompletedTicket()
    {
        IntPtr handle = ReadHandle();
        return handle != IntPtr.Zero
            ? RHIDeviceAPI.RHIDevice_GetCompletedSubmitTicket(handle)
            : 0;
    }

    public IntPtr GetSharedWin32Handle(uint index, uint generation)
    {
        IntPtr handle = ReadHandle();
        return handle != IntPtr.Zero
            ? RHIDeviceAPI.RHIDevice_GetSharedWin32Handle(handle, index, generation)
            : IntPtr.Zero;
    }

    private IntPtr ReadHandle() =>
        Interlocked.CompareExchange(ref m_NativeHandle, IntPtr.Zero, IntPtr.Zero);
}
