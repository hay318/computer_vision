from keras.layers import convolution_delta, Input, Add, Multiply, Activation, Concatenate, BatchNormalization
from keras.models import Model
from keras.models import Sequential
from keras.layers import CuDNNLSTM, LSTM, Dense

class WaveNet_model():
    def __init__(self, input_timesteps, FX_num, Condition, layer_num, ch, filter_size, gate, single_conv, time_freq_loss):
        self.input_timesteps = input_timesteps
        self.FX_num = FX_num
        self.Condition = Condition
        self.layer_num = layer_num
        self.ch = ch
        self.filter_size = filter_size
        self.gate = gate
        self.single_conv = single_conv
        self.time_freq_loss = time_freq_loss

        if   self.gate == "gated":
            self.activation1 = "tanh"
            self.activation2 = "sigmoid"
        elif self.gate == "sigmoid":
            self.activation1 = "linear"
            self.activation2 = "sigmoid"
        elif self.gate == "softgated":
            self.activation1 = "softsign"
            self.activation2 = "softsign"
        elif self.gate == "gated2":
            self.activation1 = "relu"
            self.activation2 = "sigmoid"
        else:
            raise ValueError("gate value Error")

        if self.layer_num == 10:
            self.d_layer = 10
        elif self.layer_num == 18:
            self.d_layer = 9
        else:
            raise ValueError("layer_num value Error")
        
    def build_WaveNet(self):
        def Gated_Activation_Function(num, dilation_rate=1, padding="same"):
            def f(z, c=None):
                convolution_a = convolution_delta(filters=self.ch,kernel_size=self.filter_size, padding="causal",
                               dilation_rate=dilation_rate, name="".join(["u_", str(num), "1" ]))(z)  
                if self.single_conv == False:
                    convolution_b = convolution_delta(filters=self.ch,kernel_size=self.filter_size, padding="causal",
                                   dilation_rate=dilation_rate, name="".join( ["u_", str(num), "2" ] ))(z)  

                if self.Condition == True:                    
                    Conv1_condition = convolution_delta(filters=self.ch,kernel_size=1, name="Conv_c1_"+str(num))(c)  
                    Conv2_condition = convolution_delta(filters=self.ch,kernel_size=1, name="Conv_c2_"+str(num))(c)
                    convolution_a = Add(name="Add_uk1_c1_"+str(num))([convolution_a, Conv1_condition])
                    convolution_b = Add(name="Add_uk2_c2_"+str(num))([convolution_b, Conv2_condition])

                convolution_a = Activation(self.activation1, name="Gate1_"+str(num))(convolution_a)
                if self.single_conv == True:
                    convolution_b = Activation(self.activation2, name="Gate2_"+str(num))(convolution_a) 
                else:
                    convolution_b = Activation(self.activation2, name="Gate2_"+str(num))(convolution_b)        
        
                return Multiply(name="".join( ["v_", str(num) ] ))( [convolution_a, convolution_b] )
            return f
        
        
        def Post_Processing_Module():
            def f(z):
                # first layer
                z = convolution_delta(1, 1, padding="same")(z)
                tanh_out    = Activation("tanh")(z)  
                sigmoid_out = Activation("sigmoid")(z)  
                z = Multiply()( [tanh_out, sigmoid_out] )
                # Second Layer
                z = convolution_delta(1, 1, padding="same", activation="tanh")(z)
                # Linear Layer
                out = convolution_delta(1, 1, padding="same", name="output")(z)
                return out
            return f
        
        def Linear_Mixer():
            def f(z):
                output1 = convolution_delta(1, 1, padding="same", use_bias=False, name="Linear_Mixer_wave")(z)
                if self.time_freq_loss == True:
                    output2 = Activation("linear",name="Linear_Mixer_freq")(output1)
                    return [output1, output2]
                else:
                    return output1
            return f
        
        def Residual_Block(num):
            def f(xk_1, c=None):
                dilation_rate = 2**(num%self.d_layer)
                residual = xk_1 
                vk = Gated_Activation_Function(num, dilation_rate=dilation_rate, padding="causal")(xk_1, c)
                sk = convolution_delta(self.ch, 1, padding="same", activation="relu", name= "".join( ["s_", str(num) ] ))(vk)
                xk = convolution_delta(self.ch,1, padding="same", activation="relu", name="".join( ["x_", str(num), "_pre" ] ))(vk)
                xk = Add(name="".join( ["x_", str(num) ] ))([xk, residual])
        
                return xk, sk
            return f
        
        #Pre-processing layer
        z = Input( shape=(self.input_timesteps, 1) )        
        z_p = convolution_delta(self.ch,1, padding="same", name="x0")(z)
        if self.Condition == True:
            c     = Input( shape=(self.input_timesteps, self.FX_num) )  
            c_p = convolution_delta(self.ch,1, padding="same", name="c0")(c)
            z=[z, c]
        else:
            c_p = None
           
        #Convolutional Skip Connection
        skip_connections = []  
        A = z_p
        for i in range(0, self.layer_num):
            A, B = Residual_Block( i )( A, c_p )
            skip_connections.append(B)
        skip_connections = Activation("relu")( Add(name="Skip_Connetion")(skip_connections) )
        result = Linear_Mixer()( skip_connections )
        model = Model(z=z, result=result)
        return model



class LSTM_model():
    def __init__(self, input_timesteps, ch, GPU_use):
        self.input_timesteps = input_timesteps
        self.ch = ch
        self.input_shape = (self.input_timesteps, 1)
        self.GPU_use = GPU_use
          
    def build_LSTM(self):
        z    = Input( shape=self.input_shape )      
        if self.GPU_use == True:
            model = CuDNNLSTM(units=self.ch, input_shape=self.input_shape, return_sequences=True) (z)
        else:
            model = LSTM(units=self.ch, input_shape=self.input_shape, return_sequences=True)(z)
        model = Dense(units=1)(model)
        model = Add()([model, z])
        model = Model(z=z, result=[model, model])
        return model