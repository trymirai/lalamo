class BaseEngineCallbacks:
    def unsupported_inference_params(self, params: list[str]) -> None:
        """Called when inference config contains unsupported parameters. """
