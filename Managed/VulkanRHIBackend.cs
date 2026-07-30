using System.Runtime.InteropServices;
using Arisen.Native.RHI;
using ArisenEngine.Core.RHI;
using ArisenKernel.Contracts;
using ArisenKernel.Diagnostics;
using ArisenKernel.Services;

namespace ArisenEngine.RHI.Vulkan.Native;

public sealed class VulkanRHIBackend : IRHIBackend
{
    private VulkanRHIDevice? m_DeviceService;

    public string Name => "Vulkan";

    public bool IsInitialized { get; private set; }

    public ulong Generation { get; private set; }

    [DllImport("kernel32.dll", SetLastError = true, CharSet = CharSet.Ansi)]
    private static extern IntPtr LoadLibrary(string lpFileName);

    [DllImport("kernel32.dll", CharSet = CharSet.Ansi)]
    private static extern IntPtr GetModuleHandle(string lpModuleName);

    public bool Initialize(IServiceRegistry services)
    {
        if (IsInitialized) return true;

        return InitializeCore(services, diagnosticOverride: null);
    }

    public bool Restart(IServiceRegistry services, RHIBackendRestartOptions options)
    {
        if (!Enum.IsDefined(options.DiagnosticMode))
        {
            throw new ArgumentOutOfRangeException(nameof(options));
        }

        Shutdown();
        return InitializeCore(services, options.DiagnosticMode);
    }

    private bool InitializeCore(
        IServiceRegistry services,
        RHIBackendDiagnosticMode? diagnosticOverride)
    {
        ArgumentNullException.ThrowIfNull(services);

        try
        {
            ConfigureRenderDocCapture(diagnosticOverride);

#if ARISEN_ENGINE_EDITOR
            bool initialized = InitializeEditorBackend(services);
#else
            bool initialized = InitializeRuntimeBackend(services);
#endif
            if (!initialized)
            {
                RollbackFailedInitialization();
            }

            return initialized;
        }
        catch (Exception e)
        {
            RollbackFailedInitialization();
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
                KernelLog.ErrorFormat(
                    "[VulkanRHIBackend] RHISystem.Initialize failed. Mode={0}, Reason={1}",
                    mode,
                    FormatDiagnosticReason(RHISystem.LastInitializationError));
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

            ulong nextGeneration = checked(Generation + 1);
            if (m_DeviceService == null)
            {
                var deviceService = new VulkanRHIDevice(rhiDevice.Handle, nextGeneration);
                services.RegisterService<IRHIDevice>(deviceService);
                m_DeviceService = deviceService;
            }
            else
            {
                m_DeviceService.Attach(rhiDevice.Handle, nextGeneration);
            }

            Generation = nextGeneration;
            IsInitialized = true;
            KernelLog.InfoFormat(
                "[VulkanRHIBackend] Activated IRHIDevice generation {0}. Mode={1}, Surface=0x{2:X}, Size={3}x{4}",
                Generation,
                mode,
                surfaceId,
                width,
                height);
            return true;
        }
        catch (Exception e)
        {
            KernelLog.ErrorFormat(
                "[VulkanRHIBackend] Device initialization failed. Mode={0}, Error={1}, NativeReason={2}",
                mode,
                e.Message,
                FormatDiagnosticReason(RHISystem.LastInitializationError));
            return false;
        }
    }

    private static string FormatDiagnosticReason(string? value)
    {
        return string.IsNullOrWhiteSpace(value) ? "<none>" : value;
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
        KernelLog.InfoFormat(
            "[VulkanRHIBackend] Selected adapter. Name={0}, Type={1}, {2}",
            instance.AdapterName,
            instance.AdapterTypeName,
            instance.AdapterDriverInfo);
        KernelLog.InfoFormat(
            "[VulkanRHIBackend] Enabled instance extensions: {0}",
            instance.EnabledInstanceExtensions);
        KernelLog.InfoFormat(
            "[VulkanRHIBackend] Enabled device extensions: {0}",
            instance.EnabledDeviceExtensions);
        KernelLog.InfoFormat(
            "[VulkanRHIBackend] Missing device extensions: {0}",
            instance.MissingDeviceExtensions);
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
        m_DeviceService?.Detach();
        RHISystem.Shutdown();
        IsInitialized = false;
    }

    private void RollbackFailedInitialization()
    {
        m_DeviceService?.Detach();
        IsInitialized = false;
        RHISystem.Shutdown();
    }

    private static void ConfigureRenderDocCapture(RHIBackendDiagnosticMode? diagnosticOverride)
    {
        var existingHandle = GetModuleHandle("renderdoc.dll");
        var startupMode = RenderDocStartupPolicy.Resolve(
            diagnosticOverride == RHIBackendDiagnosticMode.RenderDoc
                ? "1"
                : diagnosticOverride == RHIBackendDiagnosticMode.None
                    ? null
                    : Environment.GetEnvironmentVariable(RenderDocStartupPolicy.OptInEnvironmentVariable),
            existingHandle != IntPtr.Zero);

        if (startupMode == RenderDocStartupMode.Disabled)
        {
            Environment.SetEnvironmentVariable(
                RenderDocStartupPolicy.VulkanEnableEnvironmentVariable,
                null);
            Environment.SetEnvironmentVariable(
                RenderDocStartupPolicy.VulkanDisableEnvironmentVariable,
                "1");
            KernelLog.InfoFormat(
                "[VulkanRHIBackend] RenderDoc disabled for ordinary startup. Set {0}=1 before launch to opt in.",
                RenderDocStartupPolicy.OptInEnvironmentVariable);
            return;
        }

        if (startupMode == RenderDocStartupMode.AlreadyInjected)
        {
            Environment.SetEnvironmentVariable(
                RenderDocStartupPolicy.VulkanDisableEnvironmentVariable,
                null);
            KernelLog.Info(
                "[VulkanRHIBackend] RenderDoc was injected before backend initialization; treating the external diagnostic launch as opt-in.");
            return;
        }

        Environment.SetEnvironmentVariable(
            RenderDocStartupPolicy.VulkanDisableEnvironmentVariable,
            null);
        Environment.SetEnvironmentVariable(
            RenderDocStartupPolicy.VulkanEnableEnvironmentVariable,
            "1");

        string[] paths =
        {
            "C:\\Program Files\\RenderDoc\\renderdoc.dll",
            "C:\\renderdoc\\renderdoc.dll"
        };

        foreach (var path in paths)
        {
            if (!File.Exists(path)) continue;

            var handle = LoadLibrary(path);
            if (handle != IntPtr.Zero)
            {
                KernelLog.InfoFormat(
                    "[VulkanRHIBackend] RenderDoc explicitly enabled and preloaded from: {0}",
                    path);
                return;
            }

            KernelLog.WarningFormat(
                "[VulkanRHIBackend] RenderDoc preload failed for '{0}'. Win32Error={1}",
                path,
                Marshal.GetLastWin32Error());
        }

        KernelLog.Warning(
            "[VulkanRHIBackend] RenderDoc was requested but no preload DLL was found at a known path; the registered Vulkan implicit layer may still provide capture support.");
    }
}
