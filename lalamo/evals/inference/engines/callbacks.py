class BaseEngineCallbacks:
    def unsupported_inference_params(self, params: list[str]) -> None:
        """Called when inference config contains unsupported parameters."""

    def generation_started(self, total_rows: int) -> None:
        """Called before starting generation."""

    def generation_progress(self, rows_processed: int) -> None:
        """Called after processing each row during generation."""

    def finished_generation(self) -> None:
        """Called after all rows are processed."""
