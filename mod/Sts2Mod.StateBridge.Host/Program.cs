using Sts2Mod.StateBridge;
using Sts2Mod.StateBridge.Configuration;
using Sts2Mod.StateBridge.Logging;
using Sts2Mod.StateBridge.Providers;

var logger = new ConsoleBridgeLogger();
var requestedOptions = ParseArgs(args);
var (provider, effectiveOptions) = ProviderFactory.Create(requestedOptions, logger);
await using var bootstrap = new ModBootstrap(effectiveOptions, provider, logger);
await bootstrap.StartAsync();

var completion = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
Console.CancelKeyPress += (_, eventArgs) =>
{
    eventArgs.Cancel = true;
    completion.TrySetResult();
};
AppDomain.CurrentDomain.ProcessExit += (_, _) => completion.TrySetResult();

logger.Info("Bridge host is running. Press Ctrl+C to stop.");
await completion.Task;
await bootstrap.StopAsync();

static BridgeOptions ParseArgs(string[] args)
{
    var values = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
    for (var index = 0; index < args.Length; index += 2)
    {
        var key = args[index];
        if (!key.StartsWith("--", StringComparison.Ordinal) || index + 1 >= args.Length)
        {
            continue;
        }

        values[key[2..]] = args[index + 1];
    }

    var host = values.GetValueOrDefault("host")
        ?? Environment.GetEnvironmentVariable("STS2_BRIDGE_HOST")
        ?? "127.0.0.1";
    var portText = values.GetValueOrDefault("port")
        ?? Environment.GetEnvironmentVariable("STS2_BRIDGE_PORT");
    var writesText = values.GetValueOrDefault("enable-writes")
        ?? Environment.GetEnvironmentVariable("STS2_BRIDGE_ENABLE_WRITES");
    var readOnlyText = values.GetValueOrDefault("read-only");

    return new BridgeOptions
    {
        Host = host,
        Port = int.TryParse(portText, out var port) ? port : 17654,
        ProtocolVersion = values.GetValueOrDefault("protocol-version") ?? "0.1.0",
        ModVersion = values.GetValueOrDefault("mod-version") ?? "0.1.0",
        GameVersion = values.GetValueOrDefault("game-version") ?? "prototype",
        ProviderMode = values.GetValueOrDefault("provider-mode") ?? "fixture",
        Sts2ManagedDir = values.GetValueOrDefault("sts2-managed-dir"),
        Sts2ModLoaderDir = values.GetValueOrDefault("sts2-modloader-dir"),
        PreferRuntimeProvider = ParseBool(values.GetValueOrDefault("prefer-runtime-provider"), false),
        AllowDebugPhaseOverride = !string.Equals(values.GetValueOrDefault("allow-debug-phase-override"), "false", StringComparison.OrdinalIgnoreCase),
        ReadOnly = readOnlyText is not null
            ? !ParseBool(readOnlyText, false)
            : !ParseBool(writesText, false),
    };
}

static bool ParseBool(string? value, bool defaultValue)
{
    if (string.IsNullOrWhiteSpace(value))
    {
        return defaultValue;
    }

    return value.Trim().ToLowerInvariant() switch
    {
        "1" or "true" or "yes" or "on" => true,
        "0" or "false" or "no" or "off" => false,
        _ => defaultValue,
    };
}