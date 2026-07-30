using System;

namespace ArisenEngine.RHI.Vulkan.Native;

internal enum RenderDocStartupMode
{
    Disabled,
    PreloadRequested,
    AlreadyInjected
}

internal static class RenderDocStartupPolicy
{
    public const string OptInEnvironmentVariable = "ARISEN_ENABLE_RENDERDOC";
    public const string VulkanEnableEnvironmentVariable = "ENABLE_VULKAN_RENDERDOC_CAPTURE";
    public const string VulkanDisableEnvironmentVariable = "DISABLE_VULKAN_RENDERDOC_CAPTURE_1_42";

    public static RenderDocStartupMode Resolve(string? optInValue, bool moduleAlreadyLoaded)
    {
        if (moduleAlreadyLoaded)
        {
            return RenderDocStartupMode.AlreadyInjected;
        }

        return IsEnabledValue(optInValue)
            ? RenderDocStartupMode.PreloadRequested
            : RenderDocStartupMode.Disabled;
    }

    private static bool IsEnabledValue(string? value)
    {
        return string.Equals(value, "1", StringComparison.Ordinal) ||
               string.Equals(value, "true", StringComparison.OrdinalIgnoreCase) ||
               string.Equals(value, "yes", StringComparison.OrdinalIgnoreCase);
    }
}
