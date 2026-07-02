using System.Runtime.InteropServices;
using Arisen.Native.RHI;
using ArisenEngine.Core.RHI;
using ArisenKernel.Contracts;
using ArisenKernel.Diagnostics;
using ArisenKernel.Services;

namespace ArisenEngine.RHI.Vulkan.Native;

public sealed class VulkanRHIBackend : IRHIBackend
{
    public string Name => "Vulkan";

    public bool IsInitialized { get; private set; }

    [DllImport("kernel32.dll", SetLastError = true, CharSet = CharSet.Ansi)]
    private static extern IntPtr LoadLibrary(string lpFileName);

    [DllImport("kernel32.dll", CharSet = CharSet.Ansi)]
    private static extern IntPtr GetModuleHandle(string lpModuleName);

    public bool Initialize(IServiceRegistry services)
    {
        if (IsInitialized) return true;

        try
        {
            PreloadRenderDoc();

#if ARISEN_ENGINE_EDITOR
            return InitializeEditorBackend(services);
#else
            return InitializeRuntimeBackend(services);
#endif
        }
        catch (Exception e)
        {
            KernelLog.ErrorFormat("[VulkanRHIBackend] Graphics init failed: {0}", e.Message);
            return false;
        }
    }

    private bool InitializeEditorBackend(IServiceRegistry services)
    {
        KernelLog.Info("[VulkanRHIBackend] Initializing Vulkan for editor build. Standalone native window is not required.");
        return InitializeDeviceOnlyBackend(
            services,
            RHISystem.DefaultVirtualSurfaceID,
            width: 1920,
            height: 1080,
            mode: "EditorVirtualSurface");
    }

    private bool InitializeRuntimeBackend(IServiceRegistry services)
    {
        if (!services.TryGetService<IWindowProvider>(out var windowProvider))
        {
            KernelLog.Error("[VulkanRHIBackend] Runtime Vulkan initialization requires IWindowProvider, but none is registered.");
            return false;
        }

        var windowInfo = windowProvider.GetWindowInfo();
        if (windowInfo.SurfaceKind != WindowSurfaceKind.Win32 ||
            windowInfo.NativeHandle == IntPtr.Zero)
        {
            KernelLog.ErrorFormat(
                "[VulkanRHIBackend] Runtime Vulkan initialization requires a Win32 native window. SurfaceKind={0}, Handle=0x{1:X}, SurfaceId=0x{2:X}",
                windowInfo.SurfaceKind,
                windowInfo.NativeHandle.ToInt64(),
                windowInfo.NativeSurfaceId);
            return false;
        }

        KernelLog.InfoFormat(
            "[VulkanRHIBackend] Runtime Win32 window ready. Handle=0x{0:X}, SurfaceId=0x{1:X}, Size={2}x{3}, DpiScale={4:F2}",
            windowInfo.NativeHandle.ToInt64(),
            windowInfo.NativeSurfaceId,
            windowInfo.Width,
            windowInfo.Height,
            windowInfo.DpiScale);

        return InitializeDeviceOnlyBackend(
            services,
            windowInfo.NativeSurfaceId,
            Math.Max(1, windowInfo.Width),
            Math.Max(1, windowInfo.Height),
            mode: "RuntimeWin32Swapchain",
            initializeSwapChain: true);
    }

    private bool InitializeDeviceOnlyBackend(
        IServiceRegistry services,
        uint surfaceId,
        int width,
        int height,
        string mode,
        bool initializeSwapChain = false)
    {
        try
        {
            if (!RHISystem.Initialize(GraphicsAPI.Vulkan, validationLayer: true))
            {
                KernelLog.ErrorFormat("[VulkanRHIBackend] RHISystem.Initialize failed. Mode={0}", mode);
                return false;
            }

            var rhiDevice = RHISystem.GetOrCreateDevice(surfaceId, (uint)width, (uint)height);
            rhiDevice.SetResolution((uint)width, (uint)height);
            LogInstanceDiagnostics(rhiDevice.GetInstance(), mode);

            if (initializeSwapChain)
            {
                var rhiSurface = rhiDevice.GetSurface();
                var swapChain = rhiSurface.GetSwapChain();
                if (!swapChain.IsValid)
                {
                    KernelLog.ErrorFormat("[VulkanRHIBackend] Swapchain initialization failed. Mode={0}, Surface=0x{1:X}", mode, surfaceId);
                    return false;
                }

                KernelLog.InfoFormat("[VulkanRHIBackend] Runtime swapchain initialized. Surface=0x{0:X}, Size={1}x{2}", surfaceId, width, height);
                LogSwapChainDiagnostics(rhiDevice.GetInstance(), surfaceId);
            }

            using var registrationScope = services is ServiceRegistry registry
                ? registry.BeginPackageRegistration(VulkanRHIPackage.PackageId)
                : null;

            services.RegisterService<IRHIDevice>(new VulkanRHIDevice(rhiDevice.Handle));
            IsInitialized = true;
            KernelLog.InfoFormat(
                "[VulkanRHIBackend] Registered IRHIDevice. Mode={0}, Surface=0x{1:X}, Size={2}x{3}",
                mode,
                surfaceId,
                width,
                height);
            return true;
        }
        catch (Exception e)
        {
            KernelLog.ErrorFormat("[VulkanRHIBackend] Device initialization failed. Mode={0}, Error={1}", mode, e.Message);
            return false;
        }
    }

    private static void LogInstanceDiagnostics(RHIInstance instance, string mode)
    {
        if (!instance.IsValid)
        {
            KernelLog.WarningFormat("[VulkanRHIBackend] RHI instance diagnostics unavailable. Mode={0}", mode);
            return;
        }

        KernelLog.InfoFormat(
            "[VulkanRHIBackend] RHI instance ready. Mode={0}, Validation={1}, MaxFramesInFlight={2}, PhysicalDeviceAvailable={3}, SurfacesAvailable={4}",
            mode,
            instance.IsValidationEnabled,
            instance.MaxFramesInFlight,
            instance.IsPhysicalDeviceAvailable,
            instance.AreSurfacesAvailable);
    }

    private static void LogSwapChainDiagnostics(RHIInstance instance, uint surfaceId)
    {
        if (!instance.IsValid)
        {
            KernelLog.WarningFormat("[VulkanRHIBackend] Swapchain diagnostics unavailable. Surface=0x{0:X}", surfaceId);
            return;
        }

        var format = instance.GetSuitableSwapChainFormat(surfaceId);
        var presentMode = instance.GetSuitablePresentMode(surfaceId);
        KernelLog.InfoFormat(
            "[VulkanRHIBackend] Runtime swapchain diagnostics. Surface=0x{0:X}, Format={1}, PresentMode={2}, LinearColorSpace={3}, FifoSupported={4}",
            surfaceId,
            format,
            presentMode,
            instance.IsLinearColorSpaceSupported(surfaceId),
            instance.IsPresentModeSupported(surfaceId, EPresentMode.PRESENT_MODE_FIFO));
    }

    public void Shutdown()
    {
        RHISystem.Shutdown();
        IsInitialized = false;
    }

    private static void PreloadRenderDoc()
    {
        var handle = GetModuleHandle("renderdoc.dll");
        if (handle != IntPtr.Zero)
        {
            KernelLog.Info("[VulkanRHIBackend] RenderDoc already loaded (injected). Skipping preload.");
            return;
        }

        Environment.SetEnvironmentVariable("ENABLE_VULKAN_RENDERDOC_CAPTURE", "1");

        string[] paths =
        {
            "C:\\Program Files\\RenderDoc\\renderdoc.dll",
            "C:\\renderdoc\\renderdoc.dll"
        };

        foreach (var path in paths)
        {
            if (!File.Exists(path)) continue;

            handle = LoadLibrary(path);
            if (handle != IntPtr.Zero)
            {
                KernelLog.Info($"[VulkanRHIBackend] RenderDoc preloaded from: {path}");
                return;
            }
        }

        KernelLog.Info("[VulkanRHIBackend] RenderDoc not found. Frame capture will be unavailable.");
    }
}
