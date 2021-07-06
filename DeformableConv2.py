from keras.layers import Layer, Conv2D
from keras.engine import InputSpec
import tensorflow as tf
import keras.backend as K


class DeformableConv(Conv2D):

    def __init__(self, filters, kernel_size=(3,3), strides=1, padding='same', use_bias=False,
                 dilation_rate=(1,1), kernel_initializer='zeros', deformable_groups=1, **kwargs):
        # keep the params aligned with the conventional convs
        super(DeformableConv, self).__init__(filters=filters,
                                             kernel_size=kernel_size,
                                             strides=strides,
                                             padding=padding,
                                             use_bias=use_bias,
                                             dilation_rate=dilation_rate,
                                             kernel_initializer=kernel_initializer,
                                             **kwargs)
        self.K = kernel_size[0]*kernel_size[1] * deformable_groups
        self.use_bias = use_bias
        self.filters = filters
        self.kernel_initializer = kernel_initializer
        self.strides=strides
        self.dilation_rate = dilation_rate

    def build(self, input_shape):
        # input: feature map x, [b,h,w,c_in]
        b,h,w,c = input_shape
        # offset branch: conv2d with bias
        self.offset_kernel = self.add_weight(shape=self.kernel_size+(c,self.K*3),
                                             initializer=self.kernel_initializer,
                                             name='offset_kernel',
                                             trainable=True)
        self.offset_bias = self.add_weight(shape=(self.K*3, ),
                                           name='offset_bias',
                                           trainable=True)
        # conventional conv branch
        self.kernel = self.add_weight(shape=self.kernel_size+(c,self.filters),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      trainable=True)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.filters, ),
                                        name='bias',
                                        trainable=True)
        self.built = True

    def call(self, x):
        padding_pattern = {'same': 'SAME', 'valid': 'VALID'}
        # get offset (b,h,w,2K)
        xy_offsets = tf.nn.conv2d(x, self.kernel, self.strides, padding_pattern[self.padding])
        xy_offsets = tf.nn.bias_add(xy_offsets, self.offset_bias)
        x_offsets, y_offsets, mask = tf.split(xy_offsets, axis=-1, num_split=3)   # (b,h,w,K)
        mask = tf.sigmoid(mask)
        # do transform
        size = self.kernel_size // 2
        b,h,w,c = K.int_shape(x)
        # aggregate patches by loc (share transforms across channel): list of [b,k,k,c_in]
        patches = []
        for i in range(h):
            for j in range(w):
                # bilinear interp use corresponding offsets: list of [b,k,k,c_in]
                points = []
                for m in range(self.kernel_size):
                    for n in range(self.kernel_size):
                        x_offset = x_offsets[:,i,j,m*self.kernel_size+n]   # [b,1]
                        y_offset = y_offsets[:,i,j,m*self.kernel_size+n]
                        loc_gate = mask[:,i,j,m*self.kernel_size+n]
                        x_abs = tf.clip_by_value(j+n-size+x_offset, 0, w)   # [b,1]
                        y_abs = tf.clip_by_value(i+m-size+y_offset, 0, h)   # [b,1]
                        # lower bound
                        x0, y0 = tf.cast(x_abs, 'int32'), tf.cast(y_abs, 'int32')
                        # upper bound
                        x1, y1 = tf.clip_by_value(x_abs+1, 0, w), tf.clip_by_value(y_abs+1, 0, h)
                        points.append([x0,y0,x1,y1,x_abs,y_abs, loc_gate])       # m*n
                patch = self.bilinear_transform(x, points, m, n)    # [b,m,n,c]
                patches.append(patch)     # h*w
        # piece the patches back [b,hk,wk,c_in] and run conv with stride k: [b,h,w,c_out]
        new_map = tf.stack(patches, axis=1)   # [b,h*w,m,n,c]
        new_map = tf.reshape(new_map, shape=(b,h,w,self.kernel_size,self.kernel_size,c))
        new_map = tf.transpose(new_map, (0,1,4,2,3,5))   # [b,h,n,w,m,c]
        new_map = tf.reshape(new_map, shape=(b,h*self.kernel_size, w*self.kernel_size), c)  # (b,hk,wk,c)
        y = tf.nn.conv2d(new_map, self.kernel, strides=self.kernel_size, padding=padding_pattern[self.padding])
        if self.use_bias:
            y = tf.nn.bias_add(y, self.bias)
        return y

    def bilinear_transform(self, input, points, m, n):
        # input: [b,h,w,c]
        # points: K个[x0,y0,x1,y1,x_abs,y_abs], [b,1]
        b,h,w,c = K.int_shape(input)
        values = []
        for i in range(b):
            # take 4 nearest nodes use given indices
            for x0, y0, x1, y1, xc, yc, gate in points:
                p1 = input[b,y0[b],x0[b]]    # [c]
                w1 = (y1[b]-yc[b])*(x1[b]-xc[b]) / (y1[b]-y0[b]) / (x1[b]-x0[b])
                p2 = input[b,y0[b],x1[b]]
                w2 = (y1[b]-yc[b])*(xc[b]-x0[b]) / (y1[b]-y0[b]) / (x1[b]-x0[b])
                p3 = input[b,y1[b],x0[b]]
                w3 = (yc[b]-y0[b])*(x1[b]-xc[b]) / (y1[b]-y0[b]) / (x1[b]-x0[b])
                p4 = input[b,y1[b],x1[b]]
                w4 = (yc[b]-y0[b])*(xc[b]-x0[b]) / (y1[b]-y0[b]) / (x1[b]-x0[b])
                values.append((w1*p1+w2*p2+w3*p3+w4*p4)*gate[b])
        values = tf.stack(values, axis=0)   # [b*m*n,c]
        values = tf.reshape(values, shape=(b,m,n,c))
        return values     # [b,k,k,c]

    def compute_output_shape(self, input_shape):
        return input_shape


if __name__ == '__main__':

    pass





