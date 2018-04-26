import tensorflow as tf
import numpy

class architecture():
    def get_conv_filter(self, shape, reg, stddev):
        init = tf.random_normal_initializer(stddev=stddev)
        if reg:
            regu = tf.contrib.layers.l2_regularizer(self.wd)
            filt = tf.get_variable('filter', shape, initializer=init,regularizer=regu)
        else:
            filt = tf.get_variable('filter', shape, initializer=init)

        return filt      

    def get_bias(self, dim, init_bias, name):
        with tf.variable_scope(name):
            init = tf.constant_initializer(init_bias)
            regu = tf.contrib.layers.l2_regularizer(self.wd)
            bias = tf.get_variable('bias', dim, initializer=init, regularizer=regu)

            return bias

    def batch_norm(self, x, n_out, phase_train):
        """
        Batch normalization on convolutional maps.
        Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
        Args:
            x:           Tensor, 4D BHWD input maps
            n_out:       integer, depth of input maps
            phase_train: boolean tf.Varialbe, true indicates training phase
            scope:       string, variable scope
        Return:
            normed:      batch-normalized maps
        """
        with tf.variable_scope('bn'):

            gamma = self.get_bias(n_out, 1.0, 'gamma')
            beta = self.get_bias(n_out, 0.0, 'beta')

            batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
            ema = tf.train.ExponentialMovingAverage(decay=0.999)

            def mean_var_with_update():
                ema_apply_op = ema.apply([batch_mean, batch_var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(batch_mean), tf.identity(batch_var)

            mean, var = tf.cond(phase_train,
                                mean_var_with_update,
                                lambda: (ema.average(batch_mean), ema.average(batch_var)))
            return tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)

    def _max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
            padding='SAME', name=name)

    def _get_filter_norm(self, filt):
        eps = 1e-4
        return tf.sqrt(tf.reduce_sum(filt*filt, [0, 1, 2], keep_dims=True)+eps)

    def _get_input_norm(self, bottom, ksize, stride, pad):
        eps = 1e-4
        shape = [ksize, ksize, bottom.get_shape()[3], 1]
        filt = tf.ones(shape)
        input_norm = tf.sqrt(tf.nn.conv2d(bottom*bottom, filt, [1,stride,stride,1], padding=pad)+eps)
        return input_norm    

    def _add_orthogonal_constraint(self, filt, n_filt):
        
        filt = tf.reshape(filt, [-1, n_filt])
        inner_pro = tf.matmul(tf.transpose(filt), filt)

        loss = 2e-4*tf.nn.l2_loss(inner_pro-tf.eye(n_filt))
        tf.add_to_collection('orth_constraint', loss)

    def _conv_layer(self, bottom, ksize, n_filt, is_training, name, stride=1, bn=True, relu=True, pad='SAME', norm='cosine', reg=False, orth=False, w_norm=False):

        with tf.variable_scope(name) as scope:
            n_input = bottom.get_shape().as_list()[3]
            shape = [ksize, ksize, n_input, n_filt]
            print("shape of filter %s: %s" % (name, str(shape)))

            filt = self.get_conv_filter(shape, reg, stddev=tf.sqrt(2.0/tf.to_float(ksize*ksize*n_input)))
            conv = tf.nn.conv2d(bottom, filt, [1, stride, stride, 1], padding=pad)
            xnorm = self._get_input_norm(bottom, ksize, stride, pad)
            wnorm = self._get_filter_norm(filt)

            if w_norm == 'linear':
                conv = conv/wnorm
                conv = -0.63662*tf.acos(conv)+1
            elif w_norm == 'cosine':
                conv = conv/wnorm
            elif w_norm == 'sigmoid':
                k_value_w = 0.3
                constant_coeff_w = (1 + numpy.exp(-numpy.pi/(2*k_value_w)))/(1 - numpy.exp(-numpy.pi/(2*k_value_w)))
                conv = conv/wnorm
                conv = constant_coeff_w*(1-tf.exp(tf.acos(conv)/k_value_w-numpy.pi/(2*k_value_w)))/(1+tf.exp(tf.acos(conv)/k_value_w-numpy.pi/(2*k_value_w)))
            elif w_norm == 'none':
                pass

            if norm == 'linear':
                conv = conv/xnorm
                conv = conv/wnorm
                conv = -0.63662*tf.acos(conv)+1
            elif norm == 'cosine':
                conv = conv/xnorm
                conv = conv/wnorm
            elif norm == 'sigmoid':
                k_value = 0.3
                constant_coeff = (1 + numpy.exp(-numpy.pi/(2*k_value)))/(1 - numpy.exp(-numpy.pi/(2*k_value)))
                conv = conv/xnorm
                conv = conv/wnorm
                conv = constant_coeff*(1-tf.exp(tf.acos(conv)/k_value-numpy.pi/(2*k_value)))/(1+tf.exp(tf.acos(conv)/k_value-numpy.pi/(2*k_value)))
            elif norm == 'lr_sigmoid':
                k_value_lr = tf.get_variable('k_value_lr', n_filt,
                        initializer=tf.constant_initializer(0.7),
                        dtype=tf.float32)
                k_value_lr = tf.abs(k_value_lr) + 0.05
                constant_coeff = (1 + tf.exp(-numpy.pi/(2*k_value_lr)))/(1 - tf.exp(-numpy.pi/(2*k_value_lr)))
                conv = conv/xnorm
                conv = conv/wnorm
                conv = constant_coeff*(1-tf.exp(tf.acos(conv)/k_value_lr-numpy.pi/(2*k_value_lr)))/(1+tf.exp(tf.acos(conv)/k_value_lr-numpy.pi/(2*k_value_lr)))
            elif norm == 'none':
                pass

            if orth:
                self._add_orthogonal_constraint(filt, n_filt)
            if bn:
                conv = self.batch_norm(conv, n_filt, is_training)
            
            if relu:
                return tf.nn.relu(conv)
            else:
                return conv

    def _resnet_unit_v1(self, bottom, ksize, n_filt, is_training, name, stride, norm, reg, orth, w_norm=False):
        with tf.variable_scope(name):

            residual = self._conv_layer(bottom, ksize, n_filt, is_training, 'first', 
                                        stride, norm=norm, reg=reg, orth=orth, w_norm=w_norm)
            residual = self._conv_layer(residual, ksize, n_filt, is_training, name='second', 
                                        stride=1, relu=False, norm=norm, reg=reg, orth=orth, w_norm=w_norm)
            
            n_input = bottom.get_shape().as_list()[3]
            if n_input==n_filt:
                shortcut = bottom
            else:
                shortcut = self._conv_layer(bottom, 1, n_filt, is_training, 'shortcut', stride, bn=True, relu=False, norm='none', reg=True)
            
            return tf.nn.relu(residual + shortcut)

    # Input should be an rgb image [batch, height, width, 3]
    def build(self, rgb, n_class, is_training):        
        self.wd = 5e-4     
        ksize = 3
        n_layer = 5

        feat = (rgb - 127.5)/128.0

        #32X32        
        n_out = 96
        feat = self._conv_layer(feat, ksize, n_out, is_training, name='root', bn=True, relu=True, pad='SAME', norm='none', reg=True, orth=False)
        for i in range(n_layer):
            feat = self._resnet_unit_v1(feat, ksize, n_out, is_training, name='block1_unit'+str(i), stride=1, norm='none', reg=True, orth=False)

        #16X16
        n_out = 192
        feat = self._resnet_unit_v1(feat, ksize, n_out, is_training, name='block2_unit0', stride=2, norm='none', reg=True, orth=False)
        for i in range(1, n_layer):
            feat = self._resnet_unit_v1(feat, ksize, n_out, is_training, name='block2_unit'+str(i), stride=1, norm='none', reg=True, orth=False)

        #8X8
        n_out = 384
        feat = self._resnet_unit_v1(feat, ksize, n_out, is_training, name='block3_unit0', stride=2, norm='none', reg=True, orth=False)
        for i in range(1, n_layer):
            feat = self._resnet_unit_v1(feat, ksize, n_out, is_training, name='block3_unit'+str(i), stride=1, norm='none', reg=True, orth=False)
        
        feat = tf.nn.avg_pool(feat, [1,8,8,1], [1,1,1,1], 'VALID')
        self.score = self._conv_layer(feat, 1, n_class, is_training, "score", bn=False, relu=False, pad='VALID', norm='none', reg=True, orth=False, w_norm=False)
        self.pred = tf.squeeze(tf.argmax(self.score, axis=3))

