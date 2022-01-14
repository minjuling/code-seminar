# DataLoader

많은 양의 딥러닝 모델을 학습시키면서 많은 양의 데이터를 한번에 불러오려면 시간이 너무 오래 걸리고 RAM 까지
터지는 일이 발생한다. 데이터를 한번에 다 부르지 않고 하나씩만 불러서 쓰는 방식을 택하면 메모리가 터지지않고
모델을 돌릴 수 있다. 모든 데이터를 불러놓고 쓰는 기존의 dataset 말고 custom한 dataset을 만들어야 함
또한 길이가 변하는 input에 대해서 batch를 만들기 위해서 데이터로더에서 배치를 만드는 부분을 수정해야해서
custom dataloader를 사용해야 한다.

 
-----------------------
# Download CITYSCAPE Dataset

https://www.cityscapes-dataset.com/downloads/
