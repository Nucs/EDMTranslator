namespace EDMTranslatorTest
{
    public class FF14JaKoTranslatorConfig
    {
        public string EncoderDictDir { get; set; }
        public string ModelDir { get; set; }
    }

    public class AihubJaKoTranslatorConfig
    {
        public string EncoderDictDir { get; set; }
        public string ModelDir { get; set; }
    }

    public class JESCJaEnTranslatorConfig
    {
        public string EncoderDictDir { get; set; }
        public string ModelDir { get; set; }
    }

    /// <summary>
    /// Where the tests read their paths: <c>test_config.yaml</c> next to the test binaries, or the YAML file the
    /// <c>EDMTRANSLATOR_TEST_CONFIG</c> environment variable names (same keys), so a machine whose dictionary and models
    /// live elsewhere runs the tests without editing the committed file.
    /// </summary>
    public static class TestConfigFile
    {
        /// <summary>The environment variable that overrides the file.</summary>
        public const string EnvironmentVariable = "EDMTRANSLATOR_TEST_CONFIG";

        /// <summary>The YAML file to read: the override when it is set and not empty, else <c>test_config.yaml</c>.</summary>
        public static string Path =>
            Environment.GetEnvironmentVariable(EnvironmentVariable) is { Length: > 0 } custom ? custom : "test_config.yaml";
    }
}
