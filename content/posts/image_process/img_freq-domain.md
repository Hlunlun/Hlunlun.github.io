---
title: "Image Enhancement in Frequency Domain"
summary: "Implementation of Low-pass and High-pass Filter with Fourier Transform"
date: 2024-12-19T11:22:05+08:00
tags: ['Fourier Transform', 'Low-pass Filter', 'High-pass Filter']
---


## Q1. Remove noise from Figure 1.

### 用average filter和 median filter 分別對左圖去除雜訊，並分析和比較兩者的差別

![Figure 1.](https://prod-files-secure.s3.us-west-2.amazonaws.com/5c0abaf5-b9fa-4a9b-97e3-5ae6c274eec7/351b25a8-2b46-4fcf-a66c-c6529ca6403c/image.png)

Figure 1.

1. Average filter
    
    average filter也可以叫Smoothing Method，是用mask去對原圖做捲積，將捲積的運算的值再放到mask中心的像素，mask的size越大，整張圖會變得越平滑，如果 `k_size` =1，就跟原圖相同
    
    - `k_size`=1
    
    ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/5c0abaf5-b9fa-4a9b-97e3-5ae6c274eec7/2a933251-bdf8-4396-b33d-c95bbd9893ef/image.png)
    
    - `k_size` =5
    
    ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/5c0abaf5-b9fa-4a9b-97e3-5ae6c274eec7/93829ca6-2ec4-41d2-8f77-2a8310c3aee7/image.png)
    
    - `k_size` =9
    
    ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/5c0abaf5-b9fa-4a9b-97e3-5ae6c274eec7/d7d32223-8856-4c11-a15f-74d704424256/image.png)
    
2. Median filter
    
    找出mask到的所有數值的中間值，用這個中間值取代整個mask區塊的中間像素，也是mask的size越大，整張圖會變得越平滑
    
    - `k_size`=1
    
    ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/5c0abaf5-b9fa-4a9b-97e3-5ae6c274eec7/d1421737-d5c7-4568-8dd5-52a6ef55431d/image.png)
    
    - `k_size` =5
    
    ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/5c0abaf5-b9fa-4a9b-97e3-5ae6c274eec7/8450a1b7-bf4b-4d6a-bdc7-780913989c76/image.png)
    
    - `k_size` = 9
    
    ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/5c0abaf5-b9fa-4a9b-97e3-5ae6c274eec7/97be47a7-c10e-4d12-ad47-79de465e75c4/image.png)
    
3. Comparison
    
    
    以上改變kernel大小其實看不太出來average filter和median filter的差異，所以以下隨意添加30000黑色像素(如右圖)後再分別用這兩種方式來去除雜訊，差別就很明顯。
    
    1. Average Filter
        
        讓整張圖都變得很暗，因為它會平均像素和雜訊，造成整張圖跟添加的黑色像素融合再一起。
        
    2. Median Filter
        
        取中間值較有效的去除雜訊，因為整張圖是偏白色的，所以選取的中間值都會比較大(較接近白色)，讓整張圖沒有暗掉。
        
    
    ![image1_noise](https://prod-files-secure.s3.us-west-2.amazonaws.com/5c0abaf5-b9fa-4a9b-97e3-5ae6c274eec7/64936077-0835-4aca-a785-1587b66bac14/image.png)
    
    image1_noise
    
    ### Average Filter
    
    - `k_size` = 3
    
    ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/5c0abaf5-b9fa-4a9b-97e3-5ae6c274eec7/b721a7ad-903b-4933-a3ac-8a02dc2cf87d/image.png)
    
    - `k_size` = 5
    
    ![hw1_1_1_k-5.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/5c0abaf5-b9fa-4a9b-97e3-5ae6c274eec7/9424fc4b-495a-4e15-b547-f22ba78a76ea/hw1_1_1_k-5.jpg)
    
    - `k_size` = 9
    
    ![hw1_1_1_k-9.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/5c0abaf5-b9fa-4a9b-97e3-5ae6c274eec7/c7f395a9-87ca-42f8-9d11-f3e014e8cef0/hw1_1_1_k-9.jpg)
    
    ### Median Filter
    
    - `k_size` = 3
    
    ![hw1_1_2_k-3.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/5c0abaf5-b9fa-4a9b-97e3-5ae6c274eec7/01089548-1ef0-4ade-8146-fe4c67b99227/hw1_1_2_k-3.jpg)
    
    - `k_size` = 5
    
    ![hw1_1_2_k-5.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/5c0abaf5-b9fa-4a9b-97e3-5ae6c274eec7/76a5b5fd-26a7-4ef1-9fc4-d0a71f1a4032/hw1_1_2_k-5.jpg)
    
    - `k_size` = 9
    
    ![hw1_1_2_k-9.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/5c0abaf5-b9fa-4a9b-97e3-5ae6c274eec7/8ffe7c38-9b33-4cf6-a08b-27c5f33f4b7a/hw1_1_2_k-9.jpg)
    

## Q2. Sharp the Figure 2.

### 分別用 Sobel mask 和 Fourier transform 對左圖銳利化，並分析和比較兩者的差別

![Figure 2.](https://prod-files-secure.s3.us-west-2.amazonaws.com/5c0abaf5-b9fa-4a9b-97e3-5ae6c274eec7/b3cfce63-5b40-4aa5-a40b-05c94999d394/image.png)

Figure 2.

1. Sobel mask
    1. Sobel vs. Gaussian+Sobel: 如果有先用Gaussian去除雜訊，並且取得的邊緣pixel的值只要大於70全部調成255，會取得的乾淨俐落的邊緣，在原圖與邊緣比例皆為0.5 ( `alpha` = 0.5, `beta` = 0.5)的條件下結合兩張圖可以更明顯看出邊緣
        
        
        - Only Sobel
        
        ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/5c0abaf5-b9fa-4a9b-97e3-5ae6c274eec7/1abad72c-2bfd-47ec-acb9-a379824e5b31/image.png)
        
        - 加上原本的圖片
        
        ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/5c0abaf5-b9fa-4a9b-97e3-5ae6c274eec7/e2b29eeb-54bf-413b-ba99-d2de373db12d/image.png)
        
        - Gaussian+Sobel
        
        ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/5c0abaf5-b9fa-4a9b-97e3-5ae6c274eec7/55c15998-cedd-4406-acb9-f7c0c8abfa0d/image.png)
        
        - 加上原本的圖片
        
        ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/5c0abaf5-b9fa-4a9b-97e3-5ae6c274eec7/baf612e1-f31e-49d3-aa82-2a65ff5a44e4/image.png)
        
    2. 改變 `k_size` 肉眼來看沒甚麼差別
        
        
        - `k_size`=3
        
        ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/5c0abaf5-b9fa-4a9b-97e3-5ae6c274eec7/a2a67eec-69f0-48b2-81e9-e97788b5aaf8/image.png)
        
        - `k_size` =9
        
        ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/5c0abaf5-b9fa-4a9b-97e3-5ae6c274eec7/54d9d830-540c-4819-a72f-60273a7e2f2a/image.png)
        
    3. 改變sobel後只剩的邊緣的圖和原圖的各占比例 `alpha` 、 `beta` ，並將 `k_size` 固定為 3
        
        
        ### `alpha` = 1
        
        - `beta` = -0.5
        
        ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/5c0abaf5-b9fa-4a9b-97e3-5ae6c274eec7/e48d0b05-ed5d-45d4-8ed3-d65c90da9379/image.png)
        
        - `beta` = -1
        
        ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/5c0abaf5-b9fa-4a9b-97e3-5ae6c274eec7/fe960365-4631-417d-81eb-7e7cb2cb1f29/image.png)
        
        - `beta` = -1.5
        
        ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/5c0abaf5-b9fa-4a9b-97e3-5ae6c274eec7/8a1548b5-e946-45b9-af11-57c9779d817b/image.png)
        
        ### `beta`  =- 1
        
        - `alpha` = 0.5
        
        ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/5c0abaf5-b9fa-4a9b-97e3-5ae6c274eec7/f4ecbd93-e312-4322-aa0c-e2e47f4eecef/image.png)
        
        - `alpha` = 1
        
        ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/5c0abaf5-b9fa-4a9b-97e3-5ae6c274eec7/1ce0e2f2-4b61-44f4-83c9-db67f9a82161/image.png)
        
        - `alpha` = 1.5
        
        ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/5c0abaf5-b9fa-4a9b-97e3-5ae6c274eec7/c40070a4-39be-4dd8-bc6b-f4c5a75bbb65/image.png)
        
2. Fourier transform
    
    我用了兩種方法來盡量將邊緣強化: 
    
    1. 第一種是從phase angle去做inverse Fourier transform然後會找出邊緣，再加回原圖，效果沒有很好。基本跟原圖看不出有啥差別
        
        ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/5c0abaf5-b9fa-4a9b-97e3-5ae6c274eec7/6b0fe172-667e-4426-90f4-be7089bd77a5/image.png)
        
    2. 第二種用上課教的先算出圖片的傅立葉轉換 $F(u,v)$，然後找一個高通濾波器 $H(u,v)$ ，對 $F(u,v)$ 做捲積，最後再做 inverse Fourier transform 回去就會得到銳化的圖片。
        
        高通濾波器會呈現中間延伸某個半徑 `radius` 的範圍皆為0，其他都是1，以下分別用不同 `radius` 來比較銳化的效果
        
        - 改變 `radius` : 半徑越大，會過濾掉越多低頻資訊，所以當 `radius` = 30 就只剩邊緣這種細節的高頻資訊了，但如果太大，到最後邊緣的資訊也比較不完整了
            
            
            `radius` =0
            
            ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/5c0abaf5-b9fa-4a9b-97e3-5ae6c274eec7/698e69aa-1d3a-4d70-a080-51b0f0807c3c/fd769350-3fec-42dc-b75a-e3629fc8e12d.png)
            
            `radius` = 30
            
            ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/5c0abaf5-b9fa-4a9b-97e3-5ae6c274eec7/cfbe37fa-a781-4ad8-981c-d6834fa106d0/73ea8710-28a8-4fb1-a00a-25a42164d013.png)
            
            `radius` = 80
            
            ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/5c0abaf5-b9fa-4a9b-97e3-5ae6c274eec7/aff61795-66ae-492f-9a69-0cd3742e188b/138361a3-118e-4776-b11a-f92bb91b0f15.png)
            
            `radius` = 10
            
            ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/5c0abaf5-b9fa-4a9b-97e3-5ae6c274eec7/2ff61b6c-fa6d-49fe-b5c6-ad6a460d6559/4f20f349-6efc-40e9-8659-70b862749df0.png)
            
            `radius` = 60
            
            ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/5c0abaf5-b9fa-4a9b-97e3-5ae6c274eec7/bbcc9dd8-4b86-4af2-9f9d-62fabff35724/0ef918d5-e937-48ab-b4e7-c54f697eb7fd.png)
            
            `radius` = 100
            
            ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/5c0abaf5-b9fa-4a9b-97e3-5ae6c274eec7/a05b9d13-fe63-493b-8d06-a76e1e1393d6/abcfdfdd-bcb4-41f1-b498-096a1f2d9634.png)
            
        - 在 filter function 中隨便加上一個常數( 一半的 filter 高度)，並改變 `radius` ，也就是
            
             $G(u,v) = F(u,v) [H(u,v) + 0.5]$，半徑等於0時就看起來非常清晰了，並且隨著半徑增大，並沒有像上一種方法一樣篩出邊緣，會越接近原圖
            
            `radius` =0
            
            ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/5c0abaf5-b9fa-4a9b-97e3-5ae6c274eec7/62c2c7b5-8f4f-4c2e-8380-d9b632d73a33/034ac19f-17a9-4440-9182-83166602cee5.png)
            
            `radius` = 30
            
            ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/5c0abaf5-b9fa-4a9b-97e3-5ae6c274eec7/781e9bea-40f3-463e-8c2a-95128a034539/e7dd7b64-a107-4f3e-83e5-053bc18ea371.png)
            
            `radius` =80
            
            ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/5c0abaf5-b9fa-4a9b-97e3-5ae6c274eec7/d76d3b90-543c-4100-a0c4-38c3a62e969c/56a3790a-e974-4654-954c-0416f48b47ad.png)
            
            `radius` = 10
            
            ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/5c0abaf5-b9fa-4a9b-97e3-5ae6c274eec7/e4227da3-a705-461c-a4e2-5b07b2204265/082b9b4e-b3c3-42bb-9d74-dae3d19ef816.png)
            
            `radius` = 60
            
            ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/5c0abaf5-b9fa-4a9b-97e3-5ae6c274eec7/a8b18758-ec49-4ec1-82a4-a6a616ed8469/dcfe17b6-51d4-4356-a814-e07f278d1d8c.png)
            
            `radius` = 100
            
            ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/5c0abaf5-b9fa-4a9b-97e3-5ae6c274eec7/42aa6bf5-0794-4f5a-a35d-fe00e5ce291b/d2a6cd7c-be01-4342-84cf-55a2b2ba1996.png)
            
3. Comparison
    - 如果單純從這張青椒圖的邊緣偵測來看，Sobel 取得的邊緣是比Fourier Transform完整的
        
        
        Sobel
        
        ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/5c0abaf5-b9fa-4a9b-97e3-5ae6c274eec7/1abad72c-2bfd-47ec-acb9-a379824e5b31/image.png)
        
        Fourier Transform
        
        ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/5c0abaf5-b9fa-4a9b-97e3-5ae6c274eec7/cfbe37fa-a781-4ad8-981c-d6834fa106d0/73ea8710-28a8-4fb1-a00a-25a42164d013.png)
        
    - Sobel 銳化圖片的方式是將偵測到的邊緣在加回原圖，Fourier Transform 可以直接由高通濾波器得到邊緣相對清晰的圖
        
        
        Sobel
        
        ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/5c0abaf5-b9fa-4a9b-97e3-5ae6c274eec7/fe960365-4631-417d-81eb-7e7cb2cb1f29/image.png)
        
        Fourier Transform
        
        ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/5c0abaf5-b9fa-4a9b-97e3-5ae6c274eec7/62c2c7b5-8f4f-4c2e-8380-d9b632d73a33/034ac19f-17a9-4440-9182-83166602cee5.png)
        

## Q3. Design Low-pass Gaussian Filter

Design Gaussian filter of 3*3 mask and use this mask to low-pass filter of Figure 1.

1. Low-pass Gaussian filter
    - 中心權重高、邊緣權重低 → 保留主要像素，平滑掉高頻雜訊
    - 中心值最大為4，最接近中心的是2，其他較遠的的是1

$\begin{bmatrix}1 & 2 & 1 \\ 2 & 4 &  2\\ 1 & 2 & 1 \end{bmatrix}$

1. 用高斯低頻濾波器會將較小的像素(較暗的顏色)濾掉，使圖片少了原圖的顆粒感變得較平滑，可以去除雜訊，但是也變得比較模糊。
    
    
    - original
    
    ![Figure 1.](https://prod-files-secure.s3.us-west-2.amazonaws.com/5c0abaf5-b9fa-4a9b-97e3-5ae6c274eec7/351b25a8-2b46-4fcf-a66c-c6529ca6403c/image.png)
    
    Figure 1.
    
    - after low-pass filter
    
    ![low-pass gaussian filter](https://prod-files-secure.s3.us-west-2.amazonaws.com/5c0abaf5-b9fa-4a9b-97e3-5ae6c274eec7/d993f0ba-f509-46c9-9654-2dc7be347a98/image.png)
    
    low-pass gaussian filter
    

## Q4. Design Low-pass Fourier Filter

Design Fourier filter using q3. mask to smooth Figure 1. 

1. 如果直接把第三題的filter放到 $H(u,v)$ 中間，其他地方都是0，跟直接用第三題的 filter 去對圖像做處理肉眼其實看不出差別
    
    ```python
    H = np.zeros((rows, cols), dtype=np.float32)
    gh, gw = gaussian_filter.shape
    crow, ccol = (rows - gh) // 2, (cols - gw) // 2
    H[crow:crow+gh, ccol:ccol+gw] = gaussian_filter
    ```
    
    - original
    
    ![Figure 1.](https://prod-files-secure.s3.us-west-2.amazonaws.com/5c0abaf5-b9fa-4a9b-97e3-5ae6c274eec7/351b25a8-2b46-4fcf-a66c-c6529ca6403c/image.png)
    
    Figure 1.
    
    - After Low-pass Fourier Filter
    
    ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/5c0abaf5-b9fa-4a9b-97e3-5ae6c274eec7/131ca123-fcff-4ec1-b715-5d86f57d6893/image.png)
    
2. 用這個公式來創建low-pass filter，在丟到傅立葉轉換
    
    

## Q5. Please compute the corresponding phase angle and Fourier spectrum of Figure 3.

| 1 | 0 | 7 |
| --- | --- | --- |
| 5 | 1 | 8 |
| 4 | 0 | 9 |
1. Fourier spectrum and phase angle
    - Fourier transform
        
        
        |                     $35$ |           $-5/2+\cfrac{23\sqrt{3}}{2}j$ |           $-5/2-\cfrac{23\sqrt{3}}{2}j$ |
        | --- | --- | --- |
        |           $\cfrac{-11}{2}-\cfrac{\sqrt{3}}{2}j$ |                 $-4-\sqrt{3}j$ |                        $-1$ |
        |           $\cfrac{-11}{2}+\cfrac{\sqrt{3}}{2}j$  |                       $-1$ |                 $-4+\sqrt{3}j$ |
    - 運算結果
        
        
        Spectrum
        
        | $35$ | $20.07$ | $20.07$ |
        | --- | --- | --- |
        | $5.57$ | $4.36$ | $1$ |
        | $5.57$ | $1$ | $4.36$ |
        
        Phase Angle(rad)
        
        | 0 | $-1.446$ | $1.446$ |
        | --- | --- | --- |
        | $0.156$ | $0.409$ | $0$ |
        | $-0.156$ | $0$ | 0.409 |
    - 運算過程
        - $F(0,0)$
            
            ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/5c0abaf5-b9fa-4a9b-97e3-5ae6c274eec7/cadc134b-a004-4728-8add-0777416ff649/image.png)
            
        - $F(0,1)$
            
            ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/5c0abaf5-b9fa-4a9b-97e3-5ae6c274eec7/9543855f-9dc6-4654-ba92-9c859fa80000/image.png)
            
        - $F(0,2)$
            
            ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/5c0abaf5-b9fa-4a9b-97e3-5ae6c274eec7/3aa0ba70-87d2-45ed-8e2a-0cfaf7702129/image.png)
            
        - $F(1,0)$
            
            ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/5c0abaf5-b9fa-4a9b-97e3-5ae6c274eec7/ed61d556-fadc-4147-9842-77153f41b157/image.png)
            
        - $F(1,1)$
            
            ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/5c0abaf5-b9fa-4a9b-97e3-5ae6c274eec7/84ba55d6-42dc-40af-b036-6009f97eca38/image.png)
            
        - $F(1,2)$
            
            ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/5c0abaf5-b9fa-4a9b-97e3-5ae6c274eec7/c4474801-1eeb-4400-8a14-b1252f6c1a2d/image.png)
            
        - $F(2,0)$
            
            ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/5c0abaf5-b9fa-4a9b-97e3-5ae6c274eec7/e88f2adb-0053-486e-9889-ee83f83d1300/image.png)
            
        - $F(2,1)$
            
            ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/5c0abaf5-b9fa-4a9b-97e3-5ae6c274eec7/95d6f36a-eecf-4204-bdd5-7f921e6959f1/image.png)
            
        - $F(2,2)$
            
            ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/5c0abaf5-b9fa-4a9b-97e3-5ae6c274eec7/33074ebd-a2e0-4e73-b3cf-8d71f91dcaca/image.png)
            
    - 程式碼實現
        
        ```python
        import numpy as np
        
        # Define the given matrix f(x, y)
        f = np.array([
            [1, 0, 7],
            [5, 1, 8],
            [4, 0, 9]
        ])
        
        # Define parameters
        N = 3  # Size of the matrix
        
        # Initialize F(u, v) as a zero matrix
        F = np.zeros((N, N), dtype=complex)
        
        # Compute the 2D DFT
        for u in range(N):
            for v in range(N):
                sum_val = 0
                for x in range(N):
                    for y in range(N):
                        exp_factor = np.exp(-2j * np.pi * ((u * x + v * y) / N))
                        sum_val += f[x][y] * exp_factor
                F[u, v] = sum_val
        
        # Print matrix
        print(F)
        
        ```
        

## Reference

- [雜訊去除 — 中值濾波器 (Median filter)](https://medium.com/%E9%9B%BB%E8%85%A6%E8%A6%96%E8%A6%BA/%E5%BD%B1%E5%83%8F%E9%9B%9C%E8%A8%8A%E5%8E%BB%E9%99%A4-%E4%B8%AD%E5%80%BC%E6%BF%BE%E6%B3%A2%E5%99%A8-median-filter-e00e1ec4c86d)
- [Image Enhancement in the Frequency Domain](https://www.ee.nthu.edu.tw/clhuang/09420EE368000DIP/chapter04.pdf)