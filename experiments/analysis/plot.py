import matplotlib.pyplot as plt
import numpy as np
from plot_utils import *

if __name__ == "__main__":
    # Get Path
    args = parser()

    # Read
    all_kernels = []
    for dir in args.dirs:
        all_kernels.append(dir, collect_kernels(args))

    print(
        f"Collected {len([kernel for _, kernels in all_kernels for kernel in kernels if kernel[1] > args.min_acc])} kernels in total."
    )
    print(
        f"The kernel with smallest FLOPs is {min([kernel for _, kernels in all_kernels for kernel in kernels if kernel[1] > args.min_acc], key=lambda x:x[2])}"
    )

    # Accuracy vs FLOPs/param distribution

    # FLOPs
    all_flops_ratio = []
    if args.latency:
        all_latency_ratio = []
    all_y = []
    all_kernel_dir = []
    plt.figure(figsize=(10, 6), dpi=300)
    for i, (name, kernels) in enumerate(all_kernels):
        try:
            x, y, flops, params, hash_value, latency, kernel_dir = zip(
                *filter(lambda x: x[1] > args.min_acc, kernels)
            )
        except ValueError:
            continue

        assert (
            len(x)
            == len(y)
            == len(flops)
            == len(params)
            == len(hash_value)
            == len(latency)
        )

        all_y.extend(y)
        all_kernel_dir.extend(kernel_dir)
        flops_ratio = np.array(flops) / args.reference_flops
        all_flops_ratio.extend(list(flops_ratio))
        if args.latency:
            latency_ratio = np.array(latency) / args.reference_latency
            all_latency_ratio.extend(list(latency_ratio))

    all_flops_ratio = np.array(all_flops_ratio)
    all_y = np.array(all_y)
    if args.flops:
        plt.scatter(
            all_flops_ratio,
            all_y,
            label="FLOPs",
            s=20,
            c="#FFBE7A",
        )
        
        score = np.array([1 - all_flops_ratio, all_y]).transpose()
        pareto_mask = identify_pareto(score)
        mask = np.zeros_like(pareto_mask, np.bool_)
        for i, m in enumerate(pareto_mask):
            if m:
                print(all_kernel_dir[i])
                mask[i] = True
            else:
                
                lower_pareto = all_y[pareto_mask] <= all_y[i]
                lower_pareto[np.argmin(all_y[pareto_mask] * (~lower_pareto))] = True
                edge = np.max(all_flops_ratio[pareto_mask] * lower_pareto) if np.any(lower_pareto) else np.min(all_flops_ratio[pareto_mask])
                flops_delta = all_flops_ratio[i] - edge
                assert flops_delta >= 0
                
                upper_pareto = all_flops_ratio[pareto_mask] >= all_flops_ratio[i]
                upper_pareto[np.argmax(all_flops_ratio[pareto_mask] * (~upper_pareto))] = True
                edge = np.min(all_y[pareto_mask][upper_pareto]) if np.any(upper_pareto) else np.max(all_y[pareto_mask])
                acc_delta = - all_y[i] + edge
                assert acc_delta >= 0
                # print(f"{all_flops_ratio[i]}, {all_y[i]}, {flops_delta}, {acc_delta}")
                if flops_delta <= 0.03 or acc_delta <= 0.001:
                    print(all_kernel_dir[i])
                    mask[i] = True
        # plt.scatter(
        #     all_flops_ratio[mask],
        #     all_y[mask],
        #     label="FLOPs",
        #     s=20,
        #     c="#8ECFC9",
        # )
    
    if args.latency:
        all_latency_ratio = np.array(all_latency_ratio)
        plt.scatter(
            all_latency_ratio,
            all_y,
            label="Latency",
            s=20,
            c="#82B0D2",
        )
        if args.flops:
            for f, l, acc in zip(all_flops_ratio, all_latency_ratio, all_y):
                plt.plot([f, l], [acc, acc], c="#BEB8DC", linewidth=1.0)

        # Pareto
        score = np.array([1 - all_latency_ratio, all_y]).transpose()
        pareto_mask = identify_pareto(score)

        id = np.argsort(all_latency_ratio[pareto_mask])
        plt.plot(
            all_latency_ratio[pareto_mask][id],
            all_y[pareto_mask][id],
            label="Pareto",
            c="#8ECFC9",
            linewidth=1.3,
            # where="post",
            linestyle="--",
        )

    plt.scatter([1.0], [args.reference_acc], s=50, c="#FA7F6F", marker="^")
    plt.axhline(
        y=args.reference_acc, color="#FA7F6F", linestyle="dashed", label="acc-0"
    )
    plt.axhline(
        y=args.min_acc,
        color="#FA7F6F",
        linestyle="dashed",
        label="Min accuracy",
    )
    plt.xlabel("FLOPs and Latency (ratio to baseline)")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title(f"Search Result of {args.model}")
    plt.savefig(f"{args.output}-acc-vs-flops.png")

    # Params
    plt.figure(figsize=(10, 6), dpi=300)
    for i, (name, kernels) in enumerate(all_kernels):
        try:
            x, y, flops, params, hash_value, latency, kernel_dir = zip(
                *filter(lambda x: x[1] > args.min_acc, kernels)
            )
        except ValueError:
            continue

        assert (
            len(x)
            == len(y)
            == len(flops)
            == len(params)
            == len(hash_value)
            == len(latency)
        )

        plt.scatter(np.array(params) / args.reference_params, y, label=name, s=10)
        plt.scatter([1.0], [args.reference_acc], s=50, c="r", marker="^")
    plt.axhline(y=args.reference_acc, color="r", linestyle="dashed", label="acc-0")
    plt.axhline(
        y=args.reference_acc - 0.01, color="r", linestyle="dashed", label="acc-0.01"
    )
    plt.axhline(
        y=args.reference_acc - 0.02, color="r", linestyle="dashed", label="acc-0.02"
    )
    plt.xlabel("Params (ratio to baseline)")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(f"{args.output}-acc-vs-params.png")
