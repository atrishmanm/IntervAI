"""
models/generator/analytics.py — Training Analytics
==================================================
Comprehensive training analytics tracker (15+ metrics).

Metrics tracked:
  - Training throughput (tokens/sec, samples/sec)
  - Per-step timing breakdown (forward/backward/optimizer/data)
  - Gradient norm + variance tracking
  - Loss smoothness + convergence rate
  - GPU memory usage
  - Stability score (combined metric)
  - ETA + progress tracking
  - Dynamic dropout rate
  - Weight norm monitoring
"""

import math
import time


class TrainingAnalytics:
    """Comprehensive training analytics tracker (15+ metrics)."""

    def __init__(self, model, total_steps, stage_name="train"):
        self.model = model
        self.total_steps = total_steps
        self.stage_name = stage_name

        # Timing
        self.step_times = []
        self.fwd_times = []
        self.bwd_times = []
        self.opt_times = []
        self.data_times = []

        # Loss tracking
        self.loss_history = []
        self.smooth_loss = 0.0
        self.loss_smooth_window = 50

        # Gradient tracking
        self.grad_norms = []
        self.grad_variances = []

        # Throughput
        self.tokens_processed = 0
        self.samples_processed = 0
        self.start_time = time.time()

        # Memory
        self.peak_memory = 0.0

        # Convergence
        self.loss_plateau_count = 0
        self.best_loss = float("inf")

        # Per-epoch summaries
        self.epoch_summaries = []

    def start_step(self):
        """Call at start of each training step."""
        self._step_start = time.time()
        self._data_start = time.time()

    def end_data_loading(self):
        """Call after data loading."""
        self.data_times.append(time.time() - self._data_start)
        self._fwd_start = time.time()

    def end_forward(self):
        """Call after forward pass."""
        self.fwd_times.append(time.time() - self._fwd_start)
        self._bwd_start = time.time()

    def end_backward(self):
        """Call after backward pass."""
        self.bwd_times.append(time.time() - self._bwd_start)
        self._opt_start = time.time()

    def end_optimizer(self):
        """Call after optimizer step."""
        self.opt_times.append(time.time() - self._opt_start)
        self.step_times.append(time.time() - self._step_start)

    def update_loss(self, loss_val):
        """Update loss tracking."""
        self.loss_history.append(loss_val)
        n = min(len(self.loss_history), self.loss_smooth_window)
        self.smooth_loss = sum(self.loss_history[-n:]) / n

        if loss_val < self.best_loss:
            self.best_loss = loss_val
            self.loss_plateau_count = 0
        else:
            self.loss_plateau_count += 1

    def update_grad_norm(self, grad_norm):
        """Update gradient tracking."""
        self.grad_norms.append(grad_norm)
        if len(self.grad_norms) > 100:
            recent = self.grad_norms[-100:]
            mean_gn = sum(recent) / len(recent)
            variance = sum((g - mean_gn) ** 2 for g in recent) / len(recent)
            self.grad_variances.append(variance)

    def update_throughput(self, batch_tokens):
        """Update throughput tracking."""
        self.tokens_processed += batch_tokens
        self.samples_processed += 1

    def update_memory(self):
        """Update GPU memory tracking."""
        try:
            import torch
            if torch.cuda.is_available():
                mem = torch.cuda.max_memory_allocated() / 1e9
                self.peak_memory = max(self.peak_memory, mem)
        except Exception:
            pass

    def get_metrics(self):
        """Get all current metrics."""
        elapsed = time.time() - self.start_time

        avg_step = sum(self.step_times) / max(len(self.step_times), 1)
        avg_fwd = sum(self.fwd_times) / max(len(self.fwd_times), 1)
        avg_bwd = sum(self.bwd_times) / max(len(self.bwd_times), 1)
        avg_opt = sum(self.opt_times) / max(len(self.opt_times), 1)
        avg_data = sum(self.data_times) / max(len(self.data_times), 1)

        tok_per_sec = self.tokens_processed / max(elapsed, 0.001)
        samples_per_sec = self.samples_processed / max(elapsed, 0.001)

        steps_done = len(self.loss_history)
        if steps_done > 0 and avg_step > 0:
            eta_seconds = (self.total_steps - steps_done) * avg_step
            if eta_seconds > 3600:
                eta_str = "{:.1f}h".format(eta_seconds / 3600)
            else:
                eta_str = "{:.1f}m".format(eta_seconds / 60)
        else:
            eta_str = "calculating..."

        # Stability score (0-100)
        stability = 100.0
        if len(self.grad_norms) > 10:
            gn_recent = self.grad_norms[-100:]
            gn_mean = sum(gn_recent) / len(gn_recent)
            gn_std = (sum((g - gn_mean) ** 2 for g in gn_recent) / len(gn_recent)) ** 0.5
            if gn_mean > 0:
                cv = gn_std / gn_mean
                stability = max(0, 100 - cv * 50)
        if self.loss_plateau_count > 20:
            stability *= 0.8

        # Loss smoothness
        smoothness = 0
        if len(self.loss_history) > 10:
            recent = self.loss_history[-10:]
            loss_mean = sum(recent) / len(recent)
            loss_std = (sum((l - loss_mean) ** 2 for l in recent) / len(recent)) ** 0.5
            smoothness = max(0, 100 - loss_std * 100)

        ppl = math.exp(min(self.smooth_loss, 20)) if self.smooth_loss > 0 else 0

        return {
            "step": steps_done,
            "total_steps": self.total_steps,
            "progress_pct": steps_done / max(self.total_steps, 1) * 100,
            "eta": eta_str,
            "elapsed_min": elapsed / 60,
            "loss": self.loss_history[-1] if self.loss_history else 0,
            "smooth_loss": self.smooth_loss,
            "best_loss": self.best_loss,
            "ppl": ppl,
            "tok_per_sec": tok_per_sec,
            "samples_per_sec": samples_per_sec,
            "avg_step_ms": avg_step * 1000,
            "avg_fwd_ms": avg_fwd * 1000,
            "avg_bwd_ms": avg_bwd * 1000,
            "avg_opt_ms": avg_opt * 1000,
            "avg_data_ms": avg_data * 1000,
            "grad_norm": self.grad_norms[-1] if self.grad_norms else 0,
            "grad_norm_mean": sum(self.grad_norms[-100:]) / max(len(self.grad_norms[-100:]), 1),
            "grad_var": self.grad_variances[-1] if self.grad_variances else 0,
            "peak_memory_gb": self.peak_memory,
            "stability_score": stability,
            "smoothness_score": smoothness,
            "tokens_total": self.tokens_processed,
            "samples_total": self.samples_processed,
            "plateau_steps": self.loss_plateau_count,
        }

    def print_step_metrics(self, step_num):
        """Print compact step metrics."""
        m = self.get_metrics()
        print(
            "  step {:>5d}/{:>5d} ({:>5.1f}%) | "
            "loss {:.4f} (sm {:.4f}) | "
            "ppl {:.2f} | "
            "tok/s {:>7.0f} | "
            "grad_norm {:.3f} | "
            "stab {:.0f} | "
            "eta {} | "
            "{:.0f}ms/step".format(
                m["step"], m["total_steps"], m["progress_pct"],
                m["loss"], m["smooth_loss"],
                m["ppl"],
                m["tok_per_sec"],
                m["grad_norm"],
                m["stability_score"],
                m["eta"],
                m["avg_step_ms"],
            )
        )

    def print_epoch_summary(self, epoch, train_loss, val_metrics):
        """Print detailed epoch summary."""
        m = self.get_metrics()
        print("\n  " + "=" * 55)
        print("  EPOCH {} SUMMARY — {}".format(epoch + 1, self.stage_name.upper()))
        print("  " + "=" * 55)
        print("  Training:")
        print("    Loss:          {:.4f}".format(train_loss))
        print("    Smooth Loss:   {:.4f}".format(m["smooth_loss"]))
        print("    Best Loss:     {:.4f}".format(m["best_loss"]))
        print("    Perplexity:    {:.2f}".format(m["ppl"]))
        print("  Validation:")
        for k, v in val_metrics.items():
            if isinstance(v, float):
                print("    {}:   {:.4f}".format(k, v))
            else:
                print("    {}:   {}".format(k, v))
        print("  Performance:")
        print("    Tokens/sec:    {:.0f}".format(m["tok_per_sec"]))
        print("    Throughput:    {:.0f} samples".format(m["samples_total"]))
        print("    Peak Memory:   {:.2f} GB".format(m["peak_memory_gb"]))
        print("  Timing:")
        print("    Avg Step:      {:.0f} ms".format(m["avg_step_ms"]))
        print("    Avg Forward:   {:.0f} ms".format(m["avg_fwd_ms"]))
        print("    Avg Backward:  {:.0f} ms".format(m["avg_bwd_ms"]))
        print("    Avg Optimizer: {:.0f} ms".format(m["avg_opt_ms"]))
        print("    Avg DataLoad:  {:.0f} ms".format(m["avg_data_ms"]))
        print("  Stability:")
        print("    Grad Norm:     {:.3f} (mean: {:.3f})".format(
            m["grad_norm"], m["grad_norm_mean"]))
        print("    Grad Variance: {:.6f}".format(m["grad_var"]))
        print("    Stability:     {:.0f}/100".format(m["stability_score"]))
        print("    Smoothness:    {:.0f}/100".format(m["smoothness_score"]))
        print("    Plateau Steps: {}".format(m["plateau_steps"]))
        print("  " + "=" * 55)

        self.epoch_summaries.append(m)

    def get_final_report(self):
        """Get final training report."""
        m = self.get_metrics()
        report = {
            "stage": self.stage_name,
            "total_steps": m["step"],
            "final_loss": m["loss"],
            "best_loss": m["best_loss"],
            "final_ppl": m["ppl"],
            "avg_tok_per_sec": m["tok_per_sec"],
            "total_tokens": m["tokens_total"],
            "total_samples": m["samples_total"],
            "peak_memory_gb": m["peak_memory_gb"],
            "avg_step_ms": m["avg_step_ms"],
            "avg_fwd_ms": m["avg_fwd_ms"],
            "avg_bwd_ms": m["avg_bwd_ms"],
            "avg_opt_ms": m["avg_opt_ms"],
            "final_grad_norm": m["grad_norm"],
            "avg_grad_norm": m["grad_norm_mean"],
            "final_stability": m["stability_score"],
            "final_smoothness": m["smoothness_score"],
            "elapsed_min": m["elapsed_min"],
        }
        return report
