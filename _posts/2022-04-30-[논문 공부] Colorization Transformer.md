---
toc: true
toc_sticky: true
layout: categories
categories: 
    - "논문 공부"
tags:
    - "Computer Vision"
    - "Image Colorization"
    - "Generation Model"
---


# This post is Work In Progress.

의식의 흐름주의! 

# Colorization Transformer

오늘 읽은 논문은 

[Manoj Kumar et al., "Colorization Transformer", ICLR, 2021](https://arxiv.org/abs/2102.04432) 이다. 

Image colorization 및 생성 모델 관련 논문은 처음 보는 것이라 더 열심히 보아야 겠다. 

![title](/assets/posts/colorization_transformer/paper_title.png)

## 1. 논문의 개요

최근 Automated Image Colorization 방법들은 log-likelihood estimation에 기반한 neural generative approach가 대부분임. 


#### 1.1 Neural generative approach-based Image Colorization Methods

아래 소개된 논문들은 나중에 더 자세히 공부할 예정 우선은 메인 컨셉만 확인.

- [Amelie Royer et al. "Probabilistic Image Colorization", BMVC, 2017]("https://arxiv.org/pdf/1705.04258.pdf")

   Pixel-CNN, Pixel-CNN++ 등에 영향을 받은 방법론. 

   ![lab_img_space](/assets/posts/colorization_transformer/lab_image_space.png)
   <Figure 2.> Lab image space [[출처]](http://shutha.org/node/851)
   
   LAB image space 라고 가정했을 때, 우리는 흑백 영상의 분포를 알고 있으므로, 학습 샘플들로 Conditional distribusion ![eq001](/assets/posts/colorization_transformer/eq_001.png) 를 모델링 하는것이 목표.

   Conditional distribusion ![eq001](/assets/posts/colorization_transformer/eq_001.png)를 구하려면, 
   
   X^L은 given이고, GT(X^ab)도 아니까 모델을 찾을수 있다는 말이고 모델을 찾기 위해서 Conditional distribution을 정의함.

   여기서는 X^ab의 픽셀 값을 이전 연구처럼 deterministic 하게 추정하지 않고 샘플링하는데, 즉 VAE 처럼 일단 파라미터라이제이션 하고 hidden에서 샘플링하는데, 이때 softmax로 0~255 사이의 값(8bit)을 뽑는다. continual이 아니기 때문에 log-likelihood estimation은 멀티누이 분포를 가정하고 했다는 말이다.    
   
   Conditional distribution을 구할때는, 

   각 픽셀별로 n개의 픽셀(시행)에 대해서 [조건부 확률의 연쇄법칙(Chain rule for conditional probability)](https://blog.naver.com/PostView.naver?blogId=mykepzzang&logNo=220834907530&parentCategoryNo=&categoryNo=38&viewDate=&isShowPopularPosts=false&from=postView)에 의해서

   L 채널에서 a,b 채널로의 조건부 확률을 다음과 같이 나타낼 수 있다. 

   ![conditional_probablilty_of_Lchannel_to_ab_chnnel](/assets/posts/colorization_transformer/eq_002.png)

   테스트 시에는? 픽셀 순으로 sequantial 하게 진행.

   먼저 입력된 X_L 있으므로 ![x_1_variable](/assets/posts/colorization_transformer/eq_003.png)를 앞서 학습한 ![conditional_probability_of_Lchannel_to_ab_channel](/assets/posts/colorization_transformer/eq_004.png) 로부터 샘플링. 이후에도 연속적으로 모든 픽셀에 대해서 동일한 과정을 수행. 

   
   ![probabilistic_colorization](/assets/posts/colorization_transformer/probabilistic_colorization.png)
   <Figure 3.> Probabilistic Colorization.

   전체 네트워크는 enbedding network ![ebmedding](/assets/posts/colorization_transformer/gw.png)와 autoregressive network ![autoregress](/assets/posts/colorization_transformer/f_theta.png)로 이루어져 있다. 

   embedding netowrk ![ebmedding](/assets/posts/colorization_transformer/gw.png)는 일반적인 CNN이고, 핵심은 autoregressive network ![autoregress](/assets/posts/colorization_transformer/f_theta.png)이다. 이건 PixelCNN++이다. 

   [PixelCNN](https://arxiv.org/pdf/1606.05328.pdf) 원 논문을 살펴보면, 

   "The basic idea of the architecture is to use autoregressive connections **to model images pixel by pixel, decomposing the joint image distribution as a product of conditions**" 

   ![pixelcnn](/assets/posts/colorization_transformer/pixelcnn.png)

   커널 안에서 다음 생성될 픽셀에 대해서, 0~255 사이의 값(8bit)로 softmax prediction 수행.

   그러니까 일단 gw로 파라미터라이제이션 하고 h에서 샘플링을 픽셀 by 픽셀로 연속적으로 해서 0~256사이의 값을 가지게 추정하고

   Maximum likelihood를 향해 학습(negative loglikelihood minimization) 하겠다는것 같다. 

   crossentrophy loss 최적화 한다고 했으니 확률분포는 멀티누이 분포(Multinoulii distribution)를 가정했다느것 같다... 

   그런 식으로!!! ![eq06](/assets/posts/colorization_transformer/eq_006.png) X^L 로부터 X^ab를 prediction 한다는 것 같다. 

   그거를 굳이 수식으로 쓰면 ![eq07](/assets/posts/colorization_transformer/eq_007.png) 이렇게 된다.

   그래서 논문에서 "The current state-of-the-art in automated colorization are neural generative approaches based on
log-likelihood estimation" 라고 한것 같다. 

   
   원리는 동일한데, PixelCNN+는 나중에 더 봐야할듯. 

   일단 Colorization transformer에서는 PixelCNN이 아니라 PixelCNN+를 사용하긴 했는데, 개념은 비슷하니 다음에 본 논문 다루어볼때 하기로 하고 이쯤 넘어가자... 

   다음에 이 논문도 리뷰 해야 할듯. 


방금 전 본 것 처럼, 최근 colorization 방법들은 대부분 probabilistic model. 이전 deterministic 한 방법보다 장점있어서 널리 적용됨.

#### 1.2 Colorization Transformer 

Colorization Transformer(ColTran)도 앞서 간단히 살펴본 probabilistic corolization 모델이다.

Transformer 구조이며 [axial self-attention block](https://arxiv.org/abs/1912.12180)으로 transformer 구성. 

axial self-attention block의 장점은 global receptive field를 저렴한 비용으로 얻을 수 있다는 점이다.(자세한 이야기 논문참조)

한번에 high resolution의 gray-scale 영상을 colorization하는 것은 어렵고 연산량도 많기 때문에, 3개의 하위 sub task로 분리. 

- Coarse low resolution aotoregressive solorization

- parallel colorsuper-resolution

- spatial super-resolution 

한번에 colorization 하기 힘드니 low resolution으로 먼저 하고 super resolution한다는 뜻. 




## 2. 제안된 방법 



## 유용한 참고자료 

[오토인코더의 모든것 시리즈](https://www.youtube.com/watch?v=o_peo6U7IRM)






















