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

            if (!RHISystem.Initialize(GraphicsAPI.Vulkan, validationLayer: true))
            {
                return false;
            }

            var rhiDevice = RHISystem.GetOrCreateDevice(RHISystem.DefaultVirtualSurfaceID);
            rhiDevice.SetResolution(1920, 1080);

            using var registrationScope = services is ServiceRegistry registry
                ? registry.BeginPackageRegistration(VulkanRHIPackage.PackageId)
                : null;

            services.RegisterService<IRHIDevice>(new VulkanRHIDevice(rhiDevice.Handle));
            IsInitialized = true;
            return true;
        }
        catch (Exception e)
        {
            KernelLog.ErrorFormat("[VulkanRHIBackend] Graphics init failed: {0}", e.Message);
            return false;
        }
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
