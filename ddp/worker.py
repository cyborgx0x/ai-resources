import os

import torch
import torch.distributed as dist

# os.environ["GLOO_SOCKET_IFNAME"] = "Ethernet"py


def run(rank, size):
    """Distributed function to be implemented later."""
    print(f"Hello, world from rank {rank} out of {size} processes!")
    pass


def main():
    rank = 1
    world_size = 2
    master_addr = "172.18.0.2"
    master_port = "12345"

    # os.environ["MASTER_ADDR"] = "10.8.72.39"
    # os.environ["MASTER_PORT"] = "12345"
    dist.init_process_group(
        backend="gloo",  # Use 'nccl' for GPUs, 'gloo' or 'mpi' for CPUs or if 'nccl' isn't available
        init_method=f"tcp://{master_addr}:{master_port}",
        rank=rank,
        world_size=world_size,
    )
    # backend = "gloo"
    # dist.init_process_group(backend, rank=rank, world_size=world_size)

    # Simple "Hello, world" printout for proof of successful connection
    print(f"Hello, world from rank {rank} out of {world_size} processes!")

    # At this point, you would build and wrap your model with DDP, and begin the training loop
    # (insert model construction and training code here)

    # Cleanup - ensure to shutdown the process group
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
