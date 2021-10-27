#####################################################################################
params local rank =0
params.world_size = int(os.environ["WORLD_SIZE"]
params.n_gpu_per_node = int(os.environ["N_GPU_NODE"])
params.global_rank = int(os.environ["RANK"])
                        
if multi_gpu:
  torch.distributed.init_process_group(init_method='env://',backend='nccl')


# distributed 관련 처리 기법
# https://jjdeeplearning.tistory.com/32 참조
#https://pulsar-kkaturi.tistory.com/entry/%EB%A6%AC%EB%88%85%EC%8A%A4-%ED%84%B0%EB%AF%B8%EB%84%90%EC%97%90%EC%84%9C-%EB%94%A5%EB%9F%AC%EB%8B%9D-%ED%95%99%EC%8A%B5-%EA%B0%95%EC%A0%9C-%EC%A4%91%EB%8B%A8%ED%96%88%EC%9D%84%EB%95%8C-GPU%EC%97%90-%EB%82%A8%EC%9D%80-%EB%A9%94%EB%AA%A8%EB%A6%AC-%EC%A0%95%EB%A6%AC%ED%95%98%EB%8A%94-%EB%B0%A9%EB%B2%95
# 리눅스 터미널에서 딥러닝 학습 강제 중단했을 때 GPU에 남은 메모리 정리하는 방법
# 1. ps aux | grep python -> 딥러닝 학습을 실행시킨 python 파일의 실행 ID를 찾는다.
# 2. 찾은 아이디가 1234라면 sudo kill -9 1234 명령어 실행
# 3. nvidia-smi로 gpu 메모리 확인

# 근데 그냥 꺼주면 된다.

# https://blog.si-analytics.ai/12

# GTH 태형 전임님 말씀.

#  https://jjeamin.github.io/posts/gpus/ 참조
import torch


def setup(rank, world_size):
    # initialize the process group
    dist.init_process_group(
            backend='nccl', # nccl : NVIDIA 서버임.
            init_method='tcp://127.0.0.1:3456',
            world_size=world_size, # world size : # of processes ( single node의 경우, gpu 개수와 동일, multi node의 경우 가령 2개의 node라면, 16!)  
            rank=rank) # rank : rank of current process ( 0 ~ world_size - 1) : 
# GTH : node가 2개가 있다고 하면, 0,1,2,3,4,5,6,7 // 8,9,10,11,12,13,14,15 이렇게 되야되는데
# rank로서 global index 역할을 한다고 하면 됨. 즉 두번쨰 자리수 역할.


def cleanup():
    dist.destroy_process_group()


def main():

    n_gpus = torch.cuda.device_count()

    torch.multiprocessing.spawn(main_worker, nprocs=n_gpus, args=(n_gpus, ))

def main_worker(gpu, n_gpus):

    image_size = 224
    batch_size = 512
    num_worker = 8
    epochs = ...

    batch_size = int(batch_size / n_gpus) # 각 GPU에 들어가니까 쪼개서 넣자
    num_worker = int(num_worker / n_gpus) # 각 GPU에 들어가니까 쪼개서 넣자

    torch.distributed.init_process_group(
            backend='nccl',
            init_method='tcp://127.0.0.1:3456',
            world_size=n_gpus,
            rank=gpu)

    model = [YOUR MODEL]

    torch.cuda.set_device(gpu)
    model = model.cuda(gpu)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])

    dist.barrier()
    if cfg['load_path']:
        map_location = {'cuda:%d' % 0: 'cuda:%d' % gpu}
        model.load_state_dict(torch.load(cfg['load_path'], map_location=map_location))

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_datasets)
    train_loader = torch.utils.data.DataLoader(... , shuffle=False, sampler=train_sampler)

    optimizer = [YOUR OPTIMIZER]
    criterion = [YOUR CRITERION]

    for epoch in range(epochs):
        train()
        valid()

        if gpu == 0:
          save()

if __name__ == "__main__":
  main()

                        
