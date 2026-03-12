from datetime import datetime, timezone


class CostTracker:
    def __init__(self):
        self.session_cost = 0.0
        self.calls = []

    def log_call(self, model: str, input_tokens: int, output_tokens: int, cost: float):
        """Record a single API call."""
        self.session_cost += cost
        self.calls.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": cost,
        })

    def get_summary(self) -> dict:
        """Return aggregated stats for the session."""
        return {
            "total_calls": len(self.calls),
            "total_cost_usd": self.session_cost,
            "total_input_tokens": sum(c["input_tokens"] for c in self.calls),
            "total_output_tokens": sum(c["output_tokens"] for c in self.calls),
            "calls_detail": self.calls,
        }

    def print_summary(self):
        """Print a formatted session summary."""
        s = self.get_summary()
        print("💰 Session Summary")
        print("─────────────────")
        print(f"Calls         : {s['total_calls']}")
        print(f"Total cost    : ${s['total_cost_usd']:.6f} USD")
        print(f"Input tokens  : {s['total_input_tokens']:,}")
        print(f"Output tokens : {s['total_output_tokens']:,}")


if __name__ == "__main__":
    tracker = CostTracker()

    tracker.log_call("claude-haiku-4-5-20251001", input_tokens=500,  output_tokens=120, cost=0.000500 * 1 / 1000 + 0.000120 * 5 / 1000)
    tracker.log_call("claude-haiku-4-5-20251001", input_tokens=1200, output_tokens=300, cost=0.001200 * 1 / 1000 + 0.000300 * 5 / 1000)
    tracker.log_call("gpt-4o-mini",               input_tokens=800,  output_tokens=200, cost=0.000800 * 0.15 / 1000 + 0.000200 * 0.60 / 1000)

    tracker.print_summary()
