### opensource

    official implementation based on MXNet: https://github.com/msracver/Deformable-ConvNets
    3rd keras implementation: https://github.com/kastnerkyle/deform-conv
    3rd pytorch implementation: 
        mmdetection的mmcv库里，https://mmcv.readthedocs.io/en/latest/_modules/mmcv/ops/deform_conv.html
        centerNet2里，modified from mmcv，https://github.com/xingyizhou/CenterNet/tree/master/src/lib/models/networks/DCNv2

    deformable_groups：一个论文里没提到但是implementation都有的参数，就是字面意思，offset transform的组数


### deformable

    location specific: 
        本质上是spatial dim上的变换，所以remains same across channel
        但是相同的卷积核作用于图像的不同位置，不再是固定的感受野，不同的pixel location对应一组独立的形变参数
        RoIpooling的offset是bin location specific，即整个RoI框的平移，而非针对每个像素


### DCNv1
    
    deformable conv: 
        given input feature map: [b,h,w,c]
        先经过一个conv2d-withbias，kernel & stride & padding & diliation这些参数都保持跟conventional conv一致，通道数是2K(K是卷积核面积)，生成xy_offsets: [b,h,w,2K]
        对feature map上每个位置，都有一个唯一的kxk的核变换，将原来kxk的方形核感受野替换成一个稍微形变的核感受野
        加了offset的像素通过bilinear interpolation获取
        然后用常规卷积kernel去求加权和: [b,h,w,out_C]


    deformable RoIpooling:
        RoIpooling是有偏差的，所以后面搞了ROIAlign，这个
        given input feature map: [b,h,w,c]，以size=7为例
        先做RoIPooling，得到feature map: [b,7,7,c]
        然后经过fc，得到bin specific的变换: [b,7,7,1]
        与RoIAlign的区别就是一个是learnable的平移，一个是先验的平移


### DCNv2

    （这里我们仅关注deformable module的变化，网络整体结构&training procedure暂不讨论）

    deformable conv v2:
        还是additional conv2d-withbias branch，通道数变成3K————多了1K的mask channel，用于一致irrelevant image content
        mask的值要限定[0,1]，所以lask K channel + sigmoid


