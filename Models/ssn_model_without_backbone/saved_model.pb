�� 
�
�

B
AssignVariableOp
resource
value"dtype"
dtypetype�
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(�

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring �
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.4.12v2.4.0-49-g85c8b2a817f8��
z
dense_cc/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@* 
shared_namedense_cc/kernel
s
#dense_cc/kernel/Read/ReadVariableOpReadVariableOpdense_cc/kernel*
_output_shapes

:@*
dtype0
r
dense_cc/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_cc/bias
k
!dense_cc/bias/Read/ReadVariableOpReadVariableOpdense_cc/bias*
_output_shapes
:*
dtype0
t
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@d*
shared_namedense/kernel
m
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes

:@d*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:d*
dtype0
x
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*
shared_namedense_1/kernel
q
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes

:d*
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:*
dtype0
�
lstm_cc/lstm_cell/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	\�*)
shared_namelstm_cc/lstm_cell/kernel
�
,lstm_cc/lstm_cell/kernel/Read/ReadVariableOpReadVariableOplstm_cc/lstm_cell/kernel*
_output_shapes
:	\�*
dtype0
�
"lstm_cc/lstm_cell/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@�*3
shared_name$"lstm_cc/lstm_cell/recurrent_kernel
�
6lstm_cc/lstm_cell/recurrent_kernel/Read/ReadVariableOpReadVariableOp"lstm_cc/lstm_cell/recurrent_kernel*
_output_shapes
:	@�*
dtype0
�
lstm_cc/lstm_cell/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*'
shared_namelstm_cc/lstm_cell/bias
~
*lstm_cc/lstm_cell/bias/Read/ReadVariableOpReadVariableOplstm_cc/lstm_cell/bias*
_output_shapes	
:�*
dtype0
�
lstm/lstm_cell_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	\�*(
shared_namelstm/lstm_cell_1/kernel
�
+lstm/lstm_cell_1/kernel/Read/ReadVariableOpReadVariableOplstm/lstm_cell_1/kernel*
_output_shapes
:	\�*
dtype0
�
!lstm/lstm_cell_1/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@�*2
shared_name#!lstm/lstm_cell_1/recurrent_kernel
�
5lstm/lstm_cell_1/recurrent_kernel/Read/ReadVariableOpReadVariableOp!lstm/lstm_cell_1/recurrent_kernel*
_output_shapes
:	@�*
dtype0
�
lstm/lstm_cell_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*&
shared_namelstm/lstm_cell_1/bias
|
)lstm/lstm_cell_1/bias/Read/ReadVariableOpReadVariableOplstm/lstm_cell_1/bias*
_output_shapes	
:�*
dtype0
�
lstm_1/lstm_cell_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@�**
shared_namelstm_1/lstm_cell_2/kernel
�
-lstm_1/lstm_cell_2/kernel/Read/ReadVariableOpReadVariableOplstm_1/lstm_cell_2/kernel*
_output_shapes
:	@�*
dtype0
�
#lstm_1/lstm_cell_2/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*4
shared_name%#lstm_1/lstm_cell_2/recurrent_kernel
�
7lstm_1/lstm_cell_2/recurrent_kernel/Read/ReadVariableOpReadVariableOp#lstm_1/lstm_cell_2/recurrent_kernel* 
_output_shapes
:
��*
dtype0
�
lstm_1/lstm_cell_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*(
shared_namelstm_1/lstm_cell_2/bias
�
+lstm_1/lstm_cell_2/bias/Read/ReadVariableOpReadVariableOplstm_1/lstm_cell_2/bias*
_output_shapes	
:�*
dtype0
�
lstm_2/lstm_cell_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��**
shared_namelstm_2/lstm_cell_3/kernel
�
-lstm_2/lstm_cell_3/kernel/Read/ReadVariableOpReadVariableOplstm_2/lstm_cell_3/kernel* 
_output_shapes
:
��*
dtype0
�
#lstm_2/lstm_cell_3/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@�*4
shared_name%#lstm_2/lstm_cell_3/recurrent_kernel
�
7lstm_2/lstm_cell_3/recurrent_kernel/Read/ReadVariableOpReadVariableOp#lstm_2/lstm_cell_3/recurrent_kernel*
_output_shapes
:	@�*
dtype0
�
lstm_2/lstm_cell_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*(
shared_namelstm_2/lstm_cell_3/bias
�
+lstm_2/lstm_cell_3/bias/Read/ReadVariableOpReadVariableOplstm_2/lstm_cell_3/bias*
_output_shapes	
:�*
dtype0

NoOpNoOp
�-
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�,
value�,B�, B�,
�
lstm_cc
dense_cc
lstm1_ac
lstm2_ac
lstm3_ac
dens1_ac
	dense2_ac
	keras_api
	
signatures
l

cell

state_spec
regularization_losses
trainable_variables
	variables
	keras_api
h

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
l
cell

state_spec
regularization_losses
trainable_variables
	variables
	keras_api
l
cell

state_spec
regularization_losses
trainable_variables
 	variables
!	keras_api
l
"cell
#
state_spec
$regularization_losses
%trainable_variables
&	variables
'	keras_api
h

(kernel
)bias
*trainable_variables
+	variables
,regularization_losses
-	keras_api
h

.kernel
/bias
0trainable_variables
1	variables
2regularization_losses
3	keras_api
 
 
~

4kernel
5recurrent_kernel
6bias
7trainable_variables
8	variables
9regularization_losses
:	keras_api
 
 

40
51
62

40
51
62
�
;layer_metrics
regularization_losses
<non_trainable_variables
=metrics
trainable_variables

>layers
	variables
?layer_regularization_losses

@states
OM
VARIABLE_VALUEdense_cc/kernel*dense_cc/kernel/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEdense_cc/bias(dense_cc/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
�
Alayer_metrics
Bnon_trainable_variables
Cmetrics
trainable_variables

Dlayers
	variables
Elayer_regularization_losses
regularization_losses
~

Fkernel
Grecurrent_kernel
Hbias
Itrainable_variables
J	variables
Kregularization_losses
L	keras_api
 
 

F0
G1
H2

F0
G1
H2
�
Mlayer_metrics
regularization_losses
Nnon_trainable_variables
Ometrics
trainable_variables

Players
	variables
Qlayer_regularization_losses

Rstates
~

Skernel
Trecurrent_kernel
Ubias
Vtrainable_variables
W	variables
Xregularization_losses
Y	keras_api
 
 

S0
T1
U2

S0
T1
U2
�
Zlayer_metrics
regularization_losses
[non_trainable_variables
\metrics
trainable_variables

]layers
 	variables
^layer_regularization_losses

_states
~

`kernel
arecurrent_kernel
bbias
ctrainable_variables
d	variables
eregularization_losses
f	keras_api
 
 

`0
a1
b2

`0
a1
b2
�
glayer_metrics
$regularization_losses
hnon_trainable_variables
imetrics
%trainable_variables

jlayers
&	variables
klayer_regularization_losses

lstates
LJ
VARIABLE_VALUEdense/kernel*dens1_ac/kernel/.ATTRIBUTES/VARIABLE_VALUE
HF
VARIABLE_VALUE
dense/bias(dens1_ac/bias/.ATTRIBUTES/VARIABLE_VALUE

(0
)1

(0
)1
 
�
mlayer_metrics
nnon_trainable_variables
ometrics
*trainable_variables

players
+	variables
qlayer_regularization_losses
,regularization_losses
OM
VARIABLE_VALUEdense_1/kernel+dense2_ac/kernel/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEdense_1/bias)dense2_ac/bias/.ATTRIBUTES/VARIABLE_VALUE

.0
/1

.0
/1
 
�
rlayer_metrics
snon_trainable_variables
tmetrics
0trainable_variables

ulayers
1	variables
vlayer_regularization_losses
2regularization_losses
\Z
VARIABLE_VALUElstm_cc/lstm_cell/kernel.lstm_cc/cell/kernel/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUE"lstm_cc/lstm_cell/recurrent_kernel8lstm_cc/cell/recurrent_kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUElstm_cc/lstm_cell/bias,lstm_cc/cell/bias/.ATTRIBUTES/VARIABLE_VALUE

40
51
62

40
51
62
 
�
wlayer_metrics
xnon_trainable_variables
ymetrics
7trainable_variables

zlayers
8	variables
{layer_regularization_losses
9regularization_losses
 
 
 


0
 
 
 
 
 
 
 
\Z
VARIABLE_VALUElstm/lstm_cell_1/kernel/lstm1_ac/cell/kernel/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUE!lstm/lstm_cell_1/recurrent_kernel9lstm1_ac/cell/recurrent_kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUElstm/lstm_cell_1/bias-lstm1_ac/cell/bias/.ATTRIBUTES/VARIABLE_VALUE

F0
G1
H2

F0
G1
H2
 
�
|layer_metrics
}non_trainable_variables
~metrics
Itrainable_variables

layers
J	variables
 �layer_regularization_losses
Kregularization_losses
 
 
 

0
 
 
^\
VARIABLE_VALUElstm_1/lstm_cell_2/kernel/lstm2_ac/cell/kernel/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE#lstm_1/lstm_cell_2/recurrent_kernel9lstm2_ac/cell/recurrent_kernel/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUElstm_1/lstm_cell_2/bias-lstm2_ac/cell/bias/.ATTRIBUTES/VARIABLE_VALUE

S0
T1
U2

S0
T1
U2
 
�
�layer_metrics
�non_trainable_variables
�metrics
Vtrainable_variables
�layers
W	variables
 �layer_regularization_losses
Xregularization_losses
 
 
 

0
 
 
^\
VARIABLE_VALUElstm_2/lstm_cell_3/kernel/lstm3_ac/cell/kernel/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE#lstm_2/lstm_cell_3/recurrent_kernel9lstm3_ac/cell/recurrent_kernel/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUElstm_2/lstm_cell_3/bias-lstm3_ac/cell/bias/.ATTRIBUTES/VARIABLE_VALUE

`0
a1
b2

`0
a1
b2
 
�
�layer_metrics
�non_trainable_variables
�metrics
ctrainable_variables
�layers
d	variables
 �layer_regularization_losses
eregularization_losses
 
 
 

"0
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCallStatefulPartitionedCallsaver_filename#dense_cc/kernel/Read/ReadVariableOp!dense_cc/bias/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp,lstm_cc/lstm_cell/kernel/Read/ReadVariableOp6lstm_cc/lstm_cell/recurrent_kernel/Read/ReadVariableOp*lstm_cc/lstm_cell/bias/Read/ReadVariableOp+lstm/lstm_cell_1/kernel/Read/ReadVariableOp5lstm/lstm_cell_1/recurrent_kernel/Read/ReadVariableOp)lstm/lstm_cell_1/bias/Read/ReadVariableOp-lstm_1/lstm_cell_2/kernel/Read/ReadVariableOp7lstm_1/lstm_cell_2/recurrent_kernel/Read/ReadVariableOp+lstm_1/lstm_cell_2/bias/Read/ReadVariableOp-lstm_2/lstm_cell_3/kernel/Read/ReadVariableOp7lstm_2/lstm_cell_3/recurrent_kernel/Read/ReadVariableOp+lstm_2/lstm_cell_3/bias/Read/ReadVariableOpConst*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *)
f$R"
 __inference__traced_save_1396253
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamedense_cc/kerneldense_cc/biasdense/kernel
dense/biasdense_1/kerneldense_1/biaslstm_cc/lstm_cell/kernel"lstm_cc/lstm_cell/recurrent_kernellstm_cc/lstm_cell/biaslstm/lstm_cell_1/kernel!lstm/lstm_cell_1/recurrent_kernellstm/lstm_cell_1/biaslstm_1/lstm_cell_2/kernel#lstm_1/lstm_cell_2/recurrent_kernellstm_1/lstm_cell_2/biaslstm_2/lstm_cell_3/kernel#lstm_2/lstm_cell_3/recurrent_kernellstm_2/lstm_cell_3/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *,
f'R%
#__inference__traced_restore_1396317�
�
�
+__inference_lstm_cell_layer_call_fn_1395878

inputs
states_0
states_1
unknown
	unknown_0
	unknown_1
identity

identity_1

identity_2��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������@:���������@:���������@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_lstm_cell_layer_call_and_return_conditional_losses_13915072
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������@2

Identity�

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:���������@2

Identity_1�

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:���������@2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:���������\:���������@:���������@:::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������\
 
_user_specified_nameinputs:QM
'
_output_shapes
:���������@
"
_user_specified_name
states/0:QM
'
_output_shapes
:���������@
"
_user_specified_name
states/1
�
�
H__inference_lstm_cell_3_layer_call_and_return_conditional_losses_1393337

inputs

states
states_1"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1

identity_2��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	@�*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:����������2
add�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:���������@:���������@:���������@:���������@*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:���������@2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:���������@2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:���������@2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:���������@2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:���������@2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:���������@2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:���������@2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:���������@2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:���������@2
mul_2�
IdentityIdentity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:���������@2

Identity�

Identity_1Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:���������@2

Identity_1�

Identity_2Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:���������@2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*Y
_input_shapesH
F:����������:���������@:���������@:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������@
 
_user_specified_namestates:OK
'
_output_shapes
:���������@
 
_user_specified_namestates
�
�
while_cond_1393152
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1393152___redundant_placeholder05
1while_while_cond_1393152___redundant_placeholder15
1while_while_cond_1393152___redundant_placeholder25
1while_while_cond_1393152___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*U
_input_shapesD
B: : : : :����������:����������: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:����������:.*
(
_output_shapes
:����������:

_output_shapes
: :

_output_shapes
:
�D
�
C__inference_lstm_1_layer_call_and_return_conditional_losses_1393222

inputs
lstm_cell_2_1393140
lstm_cell_2_1393142
lstm_cell_2_1393144
identity��#lstm_cell_2/StatefulPartitionedCall�whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :�2
zeros/packed/1�
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:����������2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :�2
zeros_1/packed/1�
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:����������2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm�
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :������������������@2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1�
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2
TensorArrayV2/element_shape�
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2�
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   27
5TensorArrayUnstack/TensorListFromTensor/element_shape�
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������@*
shrink_axis_mask2
strided_slice_2�
#lstm_cell_2/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_2_1393140lstm_cell_2_1393142lstm_cell_2_1393144*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:����������:����������:����������*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_lstm_cell_2_layer_call_and_return_conditional_losses_13927272%
#lstm_cell_2/StatefulPartitionedCall�
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�����   2
TensorArrayV2_1/element_shape�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter�
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_2_1393140lstm_cell_2_1393142lstm_cell_2_1393144*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :����������:����������: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_1393153*
condR
while_cond_1393152*M
output_shapes<
:: : : : :����������:����������: : : : : *
parallel_iterations 2
while�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�����   22
0TensorArrayV2Stack/TensorListStack/element_shape�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:�������������������*
element_dtype02$
"TensorArrayV2Stack/TensorListStack�
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:�������������������2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime�
IdentityIdentitytranspose_1:y:0$^lstm_cell_2/StatefulPartitionedCall^while*
T0*5
_output_shapes#
!:�������������������2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:������������������@:::2J
#lstm_cell_2/StatefulPartitionedCall#lstm_cell_2/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :������������������@
 
_user_specified_nameinputs
�
�
while_cond_1391932
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1391932___redundant_placeholder05
1while_while_cond_1391932___redundant_placeholder15
1while_while_cond_1391932___redundant_placeholder25
1while_while_cond_1391932___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :���������@:���������@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������@:-)
'
_output_shapes
:���������@:

_output_shapes
: :

_output_shapes
:
�
�
F__inference_lstm_cell_layer_call_and_return_conditional_losses_1391474

inputs

states
states_1"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1

identity_2��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	\�*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	@�*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:����������2
add�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:���������@:���������@:���������@:���������@*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:���������@2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:���������@2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:���������@2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:���������@2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:���������@2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:���������@2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:���������@2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:���������@2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:���������@2
mul_2�
IdentityIdentity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:���������@2

Identity�

Identity_1Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:���������@2

Identity_1�

Identity_2Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:���������@2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:���������\:���������@:���������@:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:���������\
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������@
 
_user_specified_namestates:OK
'
_output_shapes
:���������@
 
_user_specified_namestates
�	
�
lstm_2_while_cond_1394208*
&lstm_2_while_lstm_2_while_loop_counter0
,lstm_2_while_lstm_2_while_maximum_iterations
lstm_2_while_placeholder
lstm_2_while_placeholder_1
lstm_2_while_placeholder_2
lstm_2_while_placeholder_3,
(lstm_2_while_less_lstm_2_strided_slice_1C
?lstm_2_while_lstm_2_while_cond_1394208___redundant_placeholder0C
?lstm_2_while_lstm_2_while_cond_1394208___redundant_placeholder1C
?lstm_2_while_lstm_2_while_cond_1394208___redundant_placeholder2C
?lstm_2_while_lstm_2_while_cond_1394208___redundant_placeholder3
lstm_2_while_identity
�
lstm_2/while/LessLesslstm_2_while_placeholder(lstm_2_while_less_lstm_2_strided_slice_1*
T0*
_output_shapes
: 2
lstm_2/while/Lessr
lstm_2/while/IdentityIdentitylstm_2/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_2/while/Identity"7
lstm_2_while_identitylstm_2/while/Identity:output:0*A
_input_shapes0
.: : : : :2@:2@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:2@:$ 

_output_shapes

:2@:

_output_shapes
: :

_output_shapes
:
�
�
F__inference_lstm_cell_layer_call_and_return_conditional_losses_1395811

inputs
states_0
states_1"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1

identity_2��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	\�*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	@�*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:����������2
add�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:���������@:���������@:���������@:���������@*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:���������@2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:���������@2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:���������@2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:���������@2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:���������@2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:���������@2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:���������@2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:���������@2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:���������@2
mul_2�
IdentityIdentity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:���������@2

Identity�

Identity_1Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:���������@2

Identity_1�

Identity_2Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:���������@2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:���������\:���������@:���������@:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:���������\
 
_user_specified_nameinputs:QM
'
_output_shapes
:���������@
"
_user_specified_name
states/0:QM
'
_output_shapes
:���������@
"
_user_specified_name
states/1
�D
�
C__inference_lstm_1_layer_call_and_return_conditional_losses_1393090

inputs
lstm_cell_2_1393008
lstm_cell_2_1393010
lstm_cell_2_1393012
identity��#lstm_cell_2/StatefulPartitionedCall�whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :�2
zeros/packed/1�
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:����������2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :�2
zeros_1/packed/1�
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:����������2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm�
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :������������������@2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1�
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2
TensorArrayV2/element_shape�
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2�
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   27
5TensorArrayUnstack/TensorListFromTensor/element_shape�
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������@*
shrink_axis_mask2
strided_slice_2�
#lstm_cell_2/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_2_1393008lstm_cell_2_1393010lstm_cell_2_1393012*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:����������:����������:����������*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_lstm_cell_2_layer_call_and_return_conditional_losses_13926942%
#lstm_cell_2/StatefulPartitionedCall�
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�����   2
TensorArrayV2_1/element_shape�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter�
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_2_1393008lstm_cell_2_1393010lstm_cell_2_1393012*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :����������:����������: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_1393021*
condR
while_cond_1393020*M
output_shapes<
:: : : : :����������:����������: : : : : *
parallel_iterations 2
while�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�����   22
0TensorArrayV2Stack/TensorListStack/element_shape�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:�������������������*
element_dtype02$
"TensorArrayV2Stack/TensorListStack�
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:�������������������2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime�
IdentityIdentitytranspose_1:y:0$^lstm_cell_2/StatefulPartitionedCall^while*
T0*5
_output_shapes#
!:�������������������2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:������������������@:::2J
#lstm_cell_2/StatefulPartitionedCall#lstm_cell_2/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :������������������@
 
_user_specified_nameinputs
�
�
F__inference_lstm_cell_layer_call_and_return_conditional_losses_1395844

inputs
states_0
states_1"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1

identity_2��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	\�*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	@�*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:����������2
add�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:���������@:���������@:���������@:���������@*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:���������@2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:���������@2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:���������@2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:���������@2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:���������@2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:���������@2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:���������@2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:���������@2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:���������@2
mul_2�
IdentityIdentity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:���������@2

Identity�

Identity_1Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:���������@2

Identity_1�

Identity_2Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:���������@2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:���������\:���������@:���������@:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:���������\
 
_user_specified_nameinputs:QM
'
_output_shapes
:���������@
"
_user_specified_name
states/0:QM
'
_output_shapes
:���������@
"
_user_specified_name
states/1
�
�
while_cond_1395517
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1395517___redundant_placeholder05
1while_while_cond_1395517___redundant_placeholder15
1while_while_cond_1395517___redundant_placeholder25
1while_while_cond_1395517___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :���������@:���������@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������@:-)
'
_output_shapes
:���������@:

_output_shapes
: :

_output_shapes
:
�@
�
while_body_1394687
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_04
0while_lstm_cell_matmul_readvariableop_resource_06
2while_lstm_cell_matmul_1_readvariableop_resource_05
1while_lstm_cell_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor2
.while_lstm_cell_matmul_readvariableop_resource4
0while_lstm_cell_matmul_1_readvariableop_resource3
/while_lstm_cell_biasadd_readvariableop_resource��&while/lstm_cell/BiasAdd/ReadVariableOp�%while/lstm_cell/MatMul/ReadVariableOp�'while/lstm_cell/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����\   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape�
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������\*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem�
%while/lstm_cell/MatMul/ReadVariableOpReadVariableOp0while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes
:	\�*
dtype02'
%while/lstm_cell/MatMul/ReadVariableOp�
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0-while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell/MatMul�
'while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp2while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes
:	@�*
dtype02)
'while/lstm_cell/MatMul_1/ReadVariableOp�
while/lstm_cell/MatMul_1MatMulwhile_placeholder_2/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell/MatMul_1�
while/lstm_cell/addAddV2 while/lstm_cell/MatMul:product:0"while/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
while/lstm_cell/add�
&while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp1while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02(
&while/lstm_cell/BiasAdd/ReadVariableOp�
while/lstm_cell/BiasAddBiasAddwhile/lstm_cell/add:z:0.while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell/BiasAddp
while/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell/Const�
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2!
while/lstm_cell/split/split_dim�
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0 while/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������@:���������@:���������@:���������@*
	num_split2
while/lstm_cell/split�
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/split:output:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell/Sigmoid�
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/split:output:1*
T0*'
_output_shapes
:���������@2
while/lstm_cell/Sigmoid_1�
while/lstm_cell/mulMulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������@2
while/lstm_cell/mul�
while/lstm_cell/ReluReluwhile/lstm_cell/split:output:2*
T0*'
_output_shapes
:���������@2
while/lstm_cell/Relu�
while/lstm_cell/mul_1Mulwhile/lstm_cell/Sigmoid:y:0"while/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell/mul_1�
while/lstm_cell/add_1AddV2while/lstm_cell/mul:z:0while/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell/add_1�
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/split:output:3*
T0*'
_output_shapes
:���������@2
while/lstm_cell/Sigmoid_2�
while/lstm_cell/Relu_1Reluwhile/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell/Relu_1�
while/lstm_cell/mul_2Mulwhile/lstm_cell/Sigmoid_2:y:0$while/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell/mul_2�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1�
while/IdentityIdentitywhile/add_1:z:0'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity�
while/Identity_1Identitywhile_while_maximum_iterations'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1�
while/Identity_2Identitywhile/add:z:0'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2�
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3�
while/Identity_4Identitywhile/lstm_cell/mul_2:z:0'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:���������@2
while/Identity_4�
while/Identity_5Identitywhile/lstm_cell/add_1:z:0'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:���������@2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"d
/while_lstm_cell_biasadd_readvariableop_resource1while_lstm_cell_biasadd_readvariableop_resource_0"f
0while_lstm_cell_matmul_1_readvariableop_resource2while_lstm_cell_matmul_1_readvariableop_resource_0"b
.while_lstm_cell_matmul_readvariableop_resource0while_lstm_cell_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :���������@:���������@: : :::2P
&while/lstm_cell/BiasAdd/ReadVariableOp&while/lstm_cell/BiasAdd/ReadVariableOp2N
%while/lstm_cell/MatMul/ReadVariableOp%while/lstm_cell/MatMul/ReadVariableOp2R
'while/lstm_cell/MatMul_1/ReadVariableOp'while/lstm_cell/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������@:-)
'
_output_shapes
:���������@:

_output_shapes
: :

_output_shapes
: 
�
�
&__inference_lstm_layer_call_fn_1395111
inputs_0
unknown
	unknown_0
	unknown_1
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_lstm_layer_call_and_return_conditional_losses_13924802
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :������������������@2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:������������������\:::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :������������������\
"
_user_specified_name
inputs/0
�
�
(__inference_lstm_2_layer_call_fn_1395778
inputs_0
unknown
	unknown_0
	unknown_1
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_lstm_2_layer_call_and_return_conditional_losses_13938322
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:�������������������:::22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:�������������������
"
_user_specified_name
inputs/0
�K
�	
lstm_2_while_body_1394209*
&lstm_2_while_lstm_2_while_loop_counter0
,lstm_2_while_lstm_2_while_maximum_iterations
lstm_2_while_placeholder
lstm_2_while_placeholder_1
lstm_2_while_placeholder_2
lstm_2_while_placeholder_3)
%lstm_2_while_lstm_2_strided_slice_1_0e
alstm_2_while_tensorarrayv2read_tensorlistgetitem_lstm_2_tensorarrayunstack_tensorlistfromtensor_0=
9lstm_2_while_lstm_cell_3_matmul_readvariableop_resource_0?
;lstm_2_while_lstm_cell_3_matmul_1_readvariableop_resource_0>
:lstm_2_while_lstm_cell_3_biasadd_readvariableop_resource_0
lstm_2_while_identity
lstm_2_while_identity_1
lstm_2_while_identity_2
lstm_2_while_identity_3
lstm_2_while_identity_4
lstm_2_while_identity_5'
#lstm_2_while_lstm_2_strided_slice_1c
_lstm_2_while_tensorarrayv2read_tensorlistgetitem_lstm_2_tensorarrayunstack_tensorlistfromtensor;
7lstm_2_while_lstm_cell_3_matmul_readvariableop_resource=
9lstm_2_while_lstm_cell_3_matmul_1_readvariableop_resource<
8lstm_2_while_lstm_cell_3_biasadd_readvariableop_resource��/lstm_2/while/lstm_cell_3/BiasAdd/ReadVariableOp�.lstm_2/while/lstm_cell_3/MatMul/ReadVariableOp�0lstm_2/while/lstm_cell_3/MatMul_1/ReadVariableOp�
>lstm_2/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"2   �   2@
>lstm_2/while/TensorArrayV2Read/TensorListGetItem/element_shape�
0lstm_2/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemalstm_2_while_tensorarrayv2read_tensorlistgetitem_lstm_2_tensorarrayunstack_tensorlistfromtensor_0lstm_2_while_placeholderGlstm_2/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
_output_shapes
:	2�*
element_dtype022
0lstm_2/while/TensorArrayV2Read/TensorListGetItem�
.lstm_2/while/lstm_cell_3/MatMul/ReadVariableOpReadVariableOp9lstm_2_while_lstm_cell_3_matmul_readvariableop_resource_0* 
_output_shapes
:
��*
dtype020
.lstm_2/while/lstm_cell_3/MatMul/ReadVariableOp�
lstm_2/while/lstm_cell_3/MatMulMatMul7lstm_2/while/TensorArrayV2Read/TensorListGetItem:item:06lstm_2/while/lstm_cell_3/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2�2!
lstm_2/while/lstm_cell_3/MatMul�
0lstm_2/while/lstm_cell_3/MatMul_1/ReadVariableOpReadVariableOp;lstm_2_while_lstm_cell_3_matmul_1_readvariableop_resource_0*
_output_shapes
:	@�*
dtype022
0lstm_2/while/lstm_cell_3/MatMul_1/ReadVariableOp�
!lstm_2/while/lstm_cell_3/MatMul_1MatMullstm_2_while_placeholder_28lstm_2/while/lstm_cell_3/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	2�2#
!lstm_2/while/lstm_cell_3/MatMul_1�
lstm_2/while/lstm_cell_3/addAddV2)lstm_2/while/lstm_cell_3/MatMul:product:0+lstm_2/while/lstm_cell_3/MatMul_1:product:0*
T0*
_output_shapes
:	2�2
lstm_2/while/lstm_cell_3/add�
/lstm_2/while/lstm_cell_3/BiasAdd/ReadVariableOpReadVariableOp:lstm_2_while_lstm_cell_3_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype021
/lstm_2/while/lstm_cell_3/BiasAdd/ReadVariableOp�
 lstm_2/while/lstm_cell_3/BiasAddBiasAdd lstm_2/while/lstm_cell_3/add:z:07lstm_2/while/lstm_cell_3/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	2�2"
 lstm_2/while/lstm_cell_3/BiasAdd�
lstm_2/while/lstm_cell_3/ConstConst*
_output_shapes
: *
dtype0*
value	B :2 
lstm_2/while/lstm_cell_3/Const�
(lstm_2/while/lstm_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2*
(lstm_2/while/lstm_cell_3/split/split_dim�
lstm_2/while/lstm_cell_3/splitSplit1lstm_2/while/lstm_cell_3/split/split_dim:output:0)lstm_2/while/lstm_cell_3/BiasAdd:output:0*
T0*<
_output_shapes*
(:2@:2@:2@:2@*
	num_split2 
lstm_2/while/lstm_cell_3/split�
 lstm_2/while/lstm_cell_3/SigmoidSigmoid'lstm_2/while/lstm_cell_3/split:output:0*
T0*
_output_shapes

:2@2"
 lstm_2/while/lstm_cell_3/Sigmoid�
"lstm_2/while/lstm_cell_3/Sigmoid_1Sigmoid'lstm_2/while/lstm_cell_3/split:output:1*
T0*
_output_shapes

:2@2$
"lstm_2/while/lstm_cell_3/Sigmoid_1�
lstm_2/while/lstm_cell_3/mulMul&lstm_2/while/lstm_cell_3/Sigmoid_1:y:0lstm_2_while_placeholder_3*
T0*
_output_shapes

:2@2
lstm_2/while/lstm_cell_3/mul�
lstm_2/while/lstm_cell_3/ReluRelu'lstm_2/while/lstm_cell_3/split:output:2*
T0*
_output_shapes

:2@2
lstm_2/while/lstm_cell_3/Relu�
lstm_2/while/lstm_cell_3/mul_1Mul$lstm_2/while/lstm_cell_3/Sigmoid:y:0+lstm_2/while/lstm_cell_3/Relu:activations:0*
T0*
_output_shapes

:2@2 
lstm_2/while/lstm_cell_3/mul_1�
lstm_2/while/lstm_cell_3/add_1AddV2 lstm_2/while/lstm_cell_3/mul:z:0"lstm_2/while/lstm_cell_3/mul_1:z:0*
T0*
_output_shapes

:2@2 
lstm_2/while/lstm_cell_3/add_1�
"lstm_2/while/lstm_cell_3/Sigmoid_2Sigmoid'lstm_2/while/lstm_cell_3/split:output:3*
T0*
_output_shapes

:2@2$
"lstm_2/while/lstm_cell_3/Sigmoid_2�
lstm_2/while/lstm_cell_3/Relu_1Relu"lstm_2/while/lstm_cell_3/add_1:z:0*
T0*
_output_shapes

:2@2!
lstm_2/while/lstm_cell_3/Relu_1�
lstm_2/while/lstm_cell_3/mul_2Mul&lstm_2/while/lstm_cell_3/Sigmoid_2:y:0-lstm_2/while/lstm_cell_3/Relu_1:activations:0*
T0*
_output_shapes

:2@2 
lstm_2/while/lstm_cell_3/mul_2�
1lstm_2/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_2_while_placeholder_1lstm_2_while_placeholder"lstm_2/while/lstm_cell_3/mul_2:z:0*
_output_shapes
: *
element_dtype023
1lstm_2/while/TensorArrayV2Write/TensorListSetItemj
lstm_2/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_2/while/add/y�
lstm_2/while/addAddV2lstm_2_while_placeholderlstm_2/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_2/while/addn
lstm_2/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_2/while/add_1/y�
lstm_2/while/add_1AddV2&lstm_2_while_lstm_2_while_loop_counterlstm_2/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_2/while/add_1�
lstm_2/while/IdentityIdentitylstm_2/while/add_1:z:00^lstm_2/while/lstm_cell_3/BiasAdd/ReadVariableOp/^lstm_2/while/lstm_cell_3/MatMul/ReadVariableOp1^lstm_2/while/lstm_cell_3/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_2/while/Identity�
lstm_2/while/Identity_1Identity,lstm_2_while_lstm_2_while_maximum_iterations0^lstm_2/while/lstm_cell_3/BiasAdd/ReadVariableOp/^lstm_2/while/lstm_cell_3/MatMul/ReadVariableOp1^lstm_2/while/lstm_cell_3/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_2/while/Identity_1�
lstm_2/while/Identity_2Identitylstm_2/while/add:z:00^lstm_2/while/lstm_cell_3/BiasAdd/ReadVariableOp/^lstm_2/while/lstm_cell_3/MatMul/ReadVariableOp1^lstm_2/while/lstm_cell_3/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_2/while/Identity_2�
lstm_2/while/Identity_3IdentityAlstm_2/while/TensorArrayV2Write/TensorListSetItem:output_handle:00^lstm_2/while/lstm_cell_3/BiasAdd/ReadVariableOp/^lstm_2/while/lstm_cell_3/MatMul/ReadVariableOp1^lstm_2/while/lstm_cell_3/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_2/while/Identity_3�
lstm_2/while/Identity_4Identity"lstm_2/while/lstm_cell_3/mul_2:z:00^lstm_2/while/lstm_cell_3/BiasAdd/ReadVariableOp/^lstm_2/while/lstm_cell_3/MatMul/ReadVariableOp1^lstm_2/while/lstm_cell_3/MatMul_1/ReadVariableOp*
T0*
_output_shapes

:2@2
lstm_2/while/Identity_4�
lstm_2/while/Identity_5Identity"lstm_2/while/lstm_cell_3/add_1:z:00^lstm_2/while/lstm_cell_3/BiasAdd/ReadVariableOp/^lstm_2/while/lstm_cell_3/MatMul/ReadVariableOp1^lstm_2/while/lstm_cell_3/MatMul_1/ReadVariableOp*
T0*
_output_shapes

:2@2
lstm_2/while/Identity_5"7
lstm_2_while_identitylstm_2/while/Identity:output:0";
lstm_2_while_identity_1 lstm_2/while/Identity_1:output:0";
lstm_2_while_identity_2 lstm_2/while/Identity_2:output:0";
lstm_2_while_identity_3 lstm_2/while/Identity_3:output:0";
lstm_2_while_identity_4 lstm_2/while/Identity_4:output:0";
lstm_2_while_identity_5 lstm_2/while/Identity_5:output:0"L
#lstm_2_while_lstm_2_strided_slice_1%lstm_2_while_lstm_2_strided_slice_1_0"v
8lstm_2_while_lstm_cell_3_biasadd_readvariableop_resource:lstm_2_while_lstm_cell_3_biasadd_readvariableop_resource_0"x
9lstm_2_while_lstm_cell_3_matmul_1_readvariableop_resource;lstm_2_while_lstm_cell_3_matmul_1_readvariableop_resource_0"t
7lstm_2_while_lstm_cell_3_matmul_readvariableop_resource9lstm_2_while_lstm_cell_3_matmul_readvariableop_resource_0"�
_lstm_2_while_tensorarrayv2read_tensorlistgetitem_lstm_2_tensorarrayunstack_tensorlistfromtensoralstm_2_while_tensorarrayv2read_tensorlistgetitem_lstm_2_tensorarrayunstack_tensorlistfromtensor_0*?
_input_shapes.
,: : : : :2@:2@: : :::2b
/lstm_2/while/lstm_cell_3/BiasAdd/ReadVariableOp/lstm_2/while/lstm_cell_3/BiasAdd/ReadVariableOp2`
.lstm_2/while/lstm_cell_3/MatMul/ReadVariableOp.lstm_2/while/lstm_cell_3/MatMul/ReadVariableOp2d
0lstm_2/while/lstm_cell_3/MatMul_1/ReadVariableOp0lstm_2/while/lstm_cell_3/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:2@:$ 

_output_shapes

:2@:

_output_shapes
: :

_output_shapes
: 
�$
�
while_body_1391801
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_lstm_cell_1391825_0
while_lstm_cell_1391827_0
while_lstm_cell_1391829_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_lstm_cell_1391825
while_lstm_cell_1391827
while_lstm_cell_1391829��'while/lstm_cell/StatefulPartitionedCall�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����\   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape�
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������\*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem�
'while/lstm_cell/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_1391825_0while_lstm_cell_1391827_0while_lstm_cell_1391829_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������@:���������@:���������@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_lstm_cell_layer_call_and_return_conditional_losses_13914742)
'while/lstm_cell/StatefulPartitionedCall�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder0while/lstm_cell/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1�
while/IdentityIdentitywhile/add_1:z:0(^while/lstm_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity�
while/Identity_1Identitywhile_while_maximum_iterations(^while/lstm_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1�
while/Identity_2Identitywhile/add:z:0(^while/lstm_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2�
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/lstm_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3�
while/Identity_4Identity0while/lstm_cell/StatefulPartitionedCall:output:1(^while/lstm_cell/StatefulPartitionedCall*
T0*'
_output_shapes
:���������@2
while/Identity_4�
while/Identity_5Identity0while/lstm_cell/StatefulPartitionedCall:output:2(^while/lstm_cell/StatefulPartitionedCall*
T0*'
_output_shapes
:���������@2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"4
while_lstm_cell_1391825while_lstm_cell_1391825_0"4
while_lstm_cell_1391827while_lstm_cell_1391827_0"4
while_lstm_cell_1391829while_lstm_cell_1391829_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :���������@:���������@: : :::2R
'while/lstm_cell/StatefulPartitionedCall'while/lstm_cell/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������@:-)
'
_output_shapes
:���������@:

_output_shapes
: :

_output_shapes
: 
�
�
+__inference_lstm_cell_layer_call_fn_1395861

inputs
states_0
states_1
unknown
	unknown_0
	unknown_1
identity

identity_1

identity_2��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������@:���������@:���������@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_lstm_cell_layer_call_and_return_conditional_losses_13914742
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������@2

Identity�

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:���������@2

Identity_1�

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:���������@2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:���������\:���������@:���������@:::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������\
 
_user_specified_nameinputs:QM
'
_output_shapes
:���������@
"
_user_specified_name
states/0:QM
'
_output_shapes
:���������@
"
_user_specified_name
states/1
�
�
H__inference_lstm_cell_1_layer_call_and_return_conditional_losses_1392117

inputs

states
states_1"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1

identity_2��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	\�*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	@�*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:����������2
add�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:���������@:���������@:���������@:���������@*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:���������@2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:���������@2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:���������@2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:���������@2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:���������@2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:���������@2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:���������@2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:���������@2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:���������@2
mul_2�
IdentityIdentity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:���������@2

Identity�

Identity_1Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:���������@2

Identity_1�

Identity_2Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:���������@2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:���������\:���������@:���������@:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:���������\
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������@
 
_user_specified_namestates:OK
'
_output_shapes
:���������@
 
_user_specified_namestates
�@
�
while_body_1394534
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_04
0while_lstm_cell_matmul_readvariableop_resource_06
2while_lstm_cell_matmul_1_readvariableop_resource_05
1while_lstm_cell_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor2
.while_lstm_cell_matmul_readvariableop_resource4
0while_lstm_cell_matmul_1_readvariableop_resource3
/while_lstm_cell_biasadd_readvariableop_resource��&while/lstm_cell/BiasAdd/ReadVariableOp�%while/lstm_cell/MatMul/ReadVariableOp�'while/lstm_cell/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����\   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape�
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������\*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem�
%while/lstm_cell/MatMul/ReadVariableOpReadVariableOp0while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes
:	\�*
dtype02'
%while/lstm_cell/MatMul/ReadVariableOp�
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0-while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell/MatMul�
'while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp2while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes
:	@�*
dtype02)
'while/lstm_cell/MatMul_1/ReadVariableOp�
while/lstm_cell/MatMul_1MatMulwhile_placeholder_2/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell/MatMul_1�
while/lstm_cell/addAddV2 while/lstm_cell/MatMul:product:0"while/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
while/lstm_cell/add�
&while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp1while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02(
&while/lstm_cell/BiasAdd/ReadVariableOp�
while/lstm_cell/BiasAddBiasAddwhile/lstm_cell/add:z:0.while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell/BiasAddp
while/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell/Const�
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2!
while/lstm_cell/split/split_dim�
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0 while/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������@:���������@:���������@:���������@*
	num_split2
while/lstm_cell/split�
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/split:output:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell/Sigmoid�
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/split:output:1*
T0*'
_output_shapes
:���������@2
while/lstm_cell/Sigmoid_1�
while/lstm_cell/mulMulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������@2
while/lstm_cell/mul�
while/lstm_cell/ReluReluwhile/lstm_cell/split:output:2*
T0*'
_output_shapes
:���������@2
while/lstm_cell/Relu�
while/lstm_cell/mul_1Mulwhile/lstm_cell/Sigmoid:y:0"while/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell/mul_1�
while/lstm_cell/add_1AddV2while/lstm_cell/mul:z:0while/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell/add_1�
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/split:output:3*
T0*'
_output_shapes
:���������@2
while/lstm_cell/Sigmoid_2�
while/lstm_cell/Relu_1Reluwhile/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell/Relu_1�
while/lstm_cell/mul_2Mulwhile/lstm_cell/Sigmoid_2:y:0$while/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell/mul_2�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1�
while/IdentityIdentitywhile/add_1:z:0'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity�
while/Identity_1Identitywhile_while_maximum_iterations'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1�
while/Identity_2Identitywhile/add:z:0'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2�
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3�
while/Identity_4Identitywhile/lstm_cell/mul_2:z:0'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:���������@2
while/Identity_4�
while/Identity_5Identitywhile/lstm_cell/add_1:z:0'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:���������@2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"d
/while_lstm_cell_biasadd_readvariableop_resource1while_lstm_cell_biasadd_readvariableop_resource_0"f
0while_lstm_cell_matmul_1_readvariableop_resource2while_lstm_cell_matmul_1_readvariableop_resource_0"b
.while_lstm_cell_matmul_readvariableop_resource0while_lstm_cell_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :���������@:���������@: : :::2P
&while/lstm_cell/BiasAdd/ReadVariableOp&while/lstm_cell/BiasAdd/ReadVariableOp2N
%while/lstm_cell/MatMul/ReadVariableOp%while/lstm_cell/MatMul/ReadVariableOp2R
'while/lstm_cell/MatMul_1/ReadVariableOp'while/lstm_cell/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������@:-)
'
_output_shapes
:���������@:

_output_shapes
: :

_output_shapes
: 
�D
�
A__inference_lstm_layer_call_and_return_conditional_losses_1392612

inputs
lstm_cell_1_1392530
lstm_cell_1_1392532
lstm_cell_1_1392534
identity��#lstm_cell_1/StatefulPartitionedCall�whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros/packed/1�
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������@2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/packed/1�
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:���������@2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm�
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :������������������\2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1�
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2
TensorArrayV2/element_shape�
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2�
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����\   27
5TensorArrayUnstack/TensorListFromTensor/element_shape�
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������\*
shrink_axis_mask2
strided_slice_2�
#lstm_cell_1/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_1_1392530lstm_cell_1_1392532lstm_cell_1_1392534*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������@:���������@:���������@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_lstm_cell_1_layer_call_and_return_conditional_losses_13921172%
#lstm_cell_1/StatefulPartitionedCall�
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   2
TensorArrayV2_1/element_shape�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter�
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_1_1392530lstm_cell_1_1392532lstm_cell_1_1392534*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������@:���������@: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_1392543*
condR
while_cond_1392542*K
output_shapes:
8: : : : :���������@:���������@: : : : : *
parallel_iterations 2
while�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   22
0TensorArrayV2Stack/TensorListStack/element_shape�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������@*
element_dtype02$
"TensorArrayV2Stack/TensorListStack�
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������@*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������@2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime�
IdentityIdentitytranspose_1:y:0$^lstm_cell_1/StatefulPartitionedCall^while*
T0*4
_output_shapes"
 :������������������@2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:������������������\:::2J
#lstm_cell_1/StatefulPartitionedCall#lstm_cell_1/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :������������������\
 
_user_specified_nameinputs
�
�
)__inference_lstm_cc_layer_call_fn_1394794
inputs_0
unknown
	unknown_0
	unknown_1
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_lstm_cc_layer_call_and_return_conditional_losses_13920022
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:������������������\:::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :������������������\
"
_user_specified_name
inputs/0
�
�
-__inference_lstm_cell_1_layer_call_fn_1395961

inputs
states_0
states_1
unknown
	unknown_0
	unknown_1
identity

identity_1

identity_2��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������@:���������@:���������@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_lstm_cell_1_layer_call_and_return_conditional_losses_13920842
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������@2

Identity�

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:���������@2

Identity_1�

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:���������@2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:���������\:���������@:���������@:::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������\
 
_user_specified_nameinputs:QM
'
_output_shapes
:���������@
"
_user_specified_name
states/0:QM
'
_output_shapes
:���������@
"
_user_specified_name
states/1
�
�
while_cond_1393630
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1393630___redundant_placeholder05
1while_while_cond_1393630___redundant_placeholder15
1while_while_cond_1393630___redundant_placeholder25
1while_while_cond_1393630___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :���������@:���������@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������@:-)
'
_output_shapes
:���������@:

_output_shapes
: :

_output_shapes
:
�J
�	
lstm_cc_while_body_1394373,
(lstm_cc_while_lstm_cc_while_loop_counter2
.lstm_cc_while_lstm_cc_while_maximum_iterations
lstm_cc_while_placeholder
lstm_cc_while_placeholder_1
lstm_cc_while_placeholder_2
lstm_cc_while_placeholder_3+
'lstm_cc_while_lstm_cc_strided_slice_1_0g
clstm_cc_while_tensorarrayv2read_tensorlistgetitem_lstm_cc_tensorarrayunstack_tensorlistfromtensor_0<
8lstm_cc_while_lstm_cell_matmul_readvariableop_resource_0>
:lstm_cc_while_lstm_cell_matmul_1_readvariableop_resource_0=
9lstm_cc_while_lstm_cell_biasadd_readvariableop_resource_0
lstm_cc_while_identity
lstm_cc_while_identity_1
lstm_cc_while_identity_2
lstm_cc_while_identity_3
lstm_cc_while_identity_4
lstm_cc_while_identity_5)
%lstm_cc_while_lstm_cc_strided_slice_1e
alstm_cc_while_tensorarrayv2read_tensorlistgetitem_lstm_cc_tensorarrayunstack_tensorlistfromtensor:
6lstm_cc_while_lstm_cell_matmul_readvariableop_resource<
8lstm_cc_while_lstm_cell_matmul_1_readvariableop_resource;
7lstm_cc_while_lstm_cell_biasadd_readvariableop_resource��.lstm_cc/while/lstm_cell/BiasAdd/ReadVariableOp�-lstm_cc/while/lstm_cell/MatMul/ReadVariableOp�/lstm_cc/while/lstm_cell/MatMul_1/ReadVariableOp�
?lstm_cc/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"2   \   2A
?lstm_cc/while/TensorArrayV2Read/TensorListGetItem/element_shape�
1lstm_cc/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_cc_while_tensorarrayv2read_tensorlistgetitem_lstm_cc_tensorarrayunstack_tensorlistfromtensor_0lstm_cc_while_placeholderHlstm_cc/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
_output_shapes

:2\*
element_dtype023
1lstm_cc/while/TensorArrayV2Read/TensorListGetItem�
-lstm_cc/while/lstm_cell/MatMul/ReadVariableOpReadVariableOp8lstm_cc_while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes
:	\�*
dtype02/
-lstm_cc/while/lstm_cell/MatMul/ReadVariableOp�
lstm_cc/while/lstm_cell/MatMulMatMul8lstm_cc/while/TensorArrayV2Read/TensorListGetItem:item:05lstm_cc/while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2�2 
lstm_cc/while/lstm_cell/MatMul�
/lstm_cc/while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp:lstm_cc_while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes
:	@�*
dtype021
/lstm_cc/while/lstm_cell/MatMul_1/ReadVariableOp�
 lstm_cc/while/lstm_cell/MatMul_1MatMullstm_cc_while_placeholder_27lstm_cc/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	2�2"
 lstm_cc/while/lstm_cell/MatMul_1�
lstm_cc/while/lstm_cell/addAddV2(lstm_cc/while/lstm_cell/MatMul:product:0*lstm_cc/while/lstm_cell/MatMul_1:product:0*
T0*
_output_shapes
:	2�2
lstm_cc/while/lstm_cell/add�
.lstm_cc/while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp9lstm_cc_while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype020
.lstm_cc/while/lstm_cell/BiasAdd/ReadVariableOp�
lstm_cc/while/lstm_cell/BiasAddBiasAddlstm_cc/while/lstm_cell/add:z:06lstm_cc/while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	2�2!
lstm_cc/while/lstm_cell/BiasAdd�
lstm_cc/while/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cc/while/lstm_cell/Const�
'lstm_cc/while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2)
'lstm_cc/while/lstm_cell/split/split_dim�
lstm_cc/while/lstm_cell/splitSplit0lstm_cc/while/lstm_cell/split/split_dim:output:0(lstm_cc/while/lstm_cell/BiasAdd:output:0*
T0*<
_output_shapes*
(:2@:2@:2@:2@*
	num_split2
lstm_cc/while/lstm_cell/split�
lstm_cc/while/lstm_cell/SigmoidSigmoid&lstm_cc/while/lstm_cell/split:output:0*
T0*
_output_shapes

:2@2!
lstm_cc/while/lstm_cell/Sigmoid�
!lstm_cc/while/lstm_cell/Sigmoid_1Sigmoid&lstm_cc/while/lstm_cell/split:output:1*
T0*
_output_shapes

:2@2#
!lstm_cc/while/lstm_cell/Sigmoid_1�
lstm_cc/while/lstm_cell/mulMul%lstm_cc/while/lstm_cell/Sigmoid_1:y:0lstm_cc_while_placeholder_3*
T0*
_output_shapes

:2@2
lstm_cc/while/lstm_cell/mul�
lstm_cc/while/lstm_cell/ReluRelu&lstm_cc/while/lstm_cell/split:output:2*
T0*
_output_shapes

:2@2
lstm_cc/while/lstm_cell/Relu�
lstm_cc/while/lstm_cell/mul_1Mul#lstm_cc/while/lstm_cell/Sigmoid:y:0*lstm_cc/while/lstm_cell/Relu:activations:0*
T0*
_output_shapes

:2@2
lstm_cc/while/lstm_cell/mul_1�
lstm_cc/while/lstm_cell/add_1AddV2lstm_cc/while/lstm_cell/mul:z:0!lstm_cc/while/lstm_cell/mul_1:z:0*
T0*
_output_shapes

:2@2
lstm_cc/while/lstm_cell/add_1�
!lstm_cc/while/lstm_cell/Sigmoid_2Sigmoid&lstm_cc/while/lstm_cell/split:output:3*
T0*
_output_shapes

:2@2#
!lstm_cc/while/lstm_cell/Sigmoid_2�
lstm_cc/while/lstm_cell/Relu_1Relu!lstm_cc/while/lstm_cell/add_1:z:0*
T0*
_output_shapes

:2@2 
lstm_cc/while/lstm_cell/Relu_1�
lstm_cc/while/lstm_cell/mul_2Mul%lstm_cc/while/lstm_cell/Sigmoid_2:y:0,lstm_cc/while/lstm_cell/Relu_1:activations:0*
T0*
_output_shapes

:2@2
lstm_cc/while/lstm_cell/mul_2�
2lstm_cc/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_cc_while_placeholder_1lstm_cc_while_placeholder!lstm_cc/while/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype024
2lstm_cc/while/TensorArrayV2Write/TensorListSetIteml
lstm_cc/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cc/while/add/y�
lstm_cc/while/addAddV2lstm_cc_while_placeholderlstm_cc/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_cc/while/addp
lstm_cc/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cc/while/add_1/y�
lstm_cc/while/add_1AddV2(lstm_cc_while_lstm_cc_while_loop_counterlstm_cc/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_cc/while/add_1�
lstm_cc/while/IdentityIdentitylstm_cc/while/add_1:z:0/^lstm_cc/while/lstm_cell/BiasAdd/ReadVariableOp.^lstm_cc/while/lstm_cell/MatMul/ReadVariableOp0^lstm_cc/while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_cc/while/Identity�
lstm_cc/while/Identity_1Identity.lstm_cc_while_lstm_cc_while_maximum_iterations/^lstm_cc/while/lstm_cell/BiasAdd/ReadVariableOp.^lstm_cc/while/lstm_cell/MatMul/ReadVariableOp0^lstm_cc/while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_cc/while/Identity_1�
lstm_cc/while/Identity_2Identitylstm_cc/while/add:z:0/^lstm_cc/while/lstm_cell/BiasAdd/ReadVariableOp.^lstm_cc/while/lstm_cell/MatMul/ReadVariableOp0^lstm_cc/while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_cc/while/Identity_2�
lstm_cc/while/Identity_3IdentityBlstm_cc/while/TensorArrayV2Write/TensorListSetItem:output_handle:0/^lstm_cc/while/lstm_cell/BiasAdd/ReadVariableOp.^lstm_cc/while/lstm_cell/MatMul/ReadVariableOp0^lstm_cc/while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_cc/while/Identity_3�
lstm_cc/while/Identity_4Identity!lstm_cc/while/lstm_cell/mul_2:z:0/^lstm_cc/while/lstm_cell/BiasAdd/ReadVariableOp.^lstm_cc/while/lstm_cell/MatMul/ReadVariableOp0^lstm_cc/while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes

:2@2
lstm_cc/while/Identity_4�
lstm_cc/while/Identity_5Identity!lstm_cc/while/lstm_cell/add_1:z:0/^lstm_cc/while/lstm_cell/BiasAdd/ReadVariableOp.^lstm_cc/while/lstm_cell/MatMul/ReadVariableOp0^lstm_cc/while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes

:2@2
lstm_cc/while/Identity_5"9
lstm_cc_while_identitylstm_cc/while/Identity:output:0"=
lstm_cc_while_identity_1!lstm_cc/while/Identity_1:output:0"=
lstm_cc_while_identity_2!lstm_cc/while/Identity_2:output:0"=
lstm_cc_while_identity_3!lstm_cc/while/Identity_3:output:0"=
lstm_cc_while_identity_4!lstm_cc/while/Identity_4:output:0"=
lstm_cc_while_identity_5!lstm_cc/while/Identity_5:output:0"P
%lstm_cc_while_lstm_cc_strided_slice_1'lstm_cc_while_lstm_cc_strided_slice_1_0"t
7lstm_cc_while_lstm_cell_biasadd_readvariableop_resource9lstm_cc_while_lstm_cell_biasadd_readvariableop_resource_0"v
8lstm_cc_while_lstm_cell_matmul_1_readvariableop_resource:lstm_cc_while_lstm_cell_matmul_1_readvariableop_resource_0"r
6lstm_cc_while_lstm_cell_matmul_readvariableop_resource8lstm_cc_while_lstm_cell_matmul_readvariableop_resource_0"�
alstm_cc_while_tensorarrayv2read_tensorlistgetitem_lstm_cc_tensorarrayunstack_tensorlistfromtensorclstm_cc_while_tensorarrayv2read_tensorlistgetitem_lstm_cc_tensorarrayunstack_tensorlistfromtensor_0*?
_input_shapes.
,: : : : :2@:2@: : :::2`
.lstm_cc/while/lstm_cell/BiasAdd/ReadVariableOp.lstm_cc/while/lstm_cell/BiasAdd/ReadVariableOp2^
-lstm_cc/while/lstm_cell/MatMul/ReadVariableOp-lstm_cc/while/lstm_cell/MatMul/ReadVariableOp2b
/lstm_cc/while/lstm_cell/MatMul_1/ReadVariableOp/lstm_cc/while/lstm_cell/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:2@:$ 

_output_shapes

:2@:

_output_shapes
: :

_output_shapes
: 
�%
�
while_body_1393153
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_lstm_cell_2_1393177_0
while_lstm_cell_2_1393179_0
while_lstm_cell_2_1393181_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_lstm_cell_2_1393177
while_lstm_cell_2_1393179
while_lstm_cell_2_1393181��)while/lstm_cell_2/StatefulPartitionedCall�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape�
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������@*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem�
)while/lstm_cell_2/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_2_1393177_0while_lstm_cell_2_1393179_0while_lstm_cell_2_1393181_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:����������:����������:����������*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_lstm_cell_2_layer_call_and_return_conditional_losses_13927272+
)while/lstm_cell_2/StatefulPartitionedCall�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/lstm_cell_2/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1�
while/IdentityIdentitywhile/add_1:z:0*^while/lstm_cell_2/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity�
while/Identity_1Identitywhile_while_maximum_iterations*^while/lstm_cell_2/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1�
while/Identity_2Identitywhile/add:z:0*^while/lstm_cell_2/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2�
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*^while/lstm_cell_2/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3�
while/Identity_4Identity2while/lstm_cell_2/StatefulPartitionedCall:output:1*^while/lstm_cell_2/StatefulPartitionedCall*
T0*(
_output_shapes
:����������2
while/Identity_4�
while/Identity_5Identity2while/lstm_cell_2/StatefulPartitionedCall:output:2*^while/lstm_cell_2/StatefulPartitionedCall*
T0*(
_output_shapes
:����������2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"8
while_lstm_cell_2_1393177while_lstm_cell_2_1393177_0"8
while_lstm_cell_2_1393179while_lstm_cell_2_1393179_0"8
while_lstm_cell_2_1393181while_lstm_cell_2_1393181_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :����������:����������: : :::2V
)while/lstm_cell_2/StatefulPartitionedCall)while/lstm_cell_2/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:����������:.*
(
_output_shapes
:����������:

_output_shapes
: :

_output_shapes
: 
�
�
while_cond_1392410
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1392410___redundant_placeholder05
1while_while_cond_1392410___redundant_placeholder15
1while_while_cond_1392410___redundant_placeholder25
1while_while_cond_1392410___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :���������@:���������@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������@:-)
'
_output_shapes
:���������@:

_output_shapes
: :

_output_shapes
:
�
�
H__inference_lstm_cell_1_layer_call_and_return_conditional_losses_1395944

inputs
states_0
states_1"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1

identity_2��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	\�*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	@�*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:����������2
add�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:���������@:���������@:���������@:���������@*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:���������@2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:���������@2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:���������@2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:���������@2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:���������@2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:���������@2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:���������@2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:���������@2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:���������@2
mul_2�
IdentityIdentity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:���������@2

Identity�

Identity_1Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:���������@2

Identity_1�

Identity_2Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:���������@2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:���������\:���������@:���������@:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:���������\
 
_user_specified_nameinputs:QM
'
_output_shapes
:���������@
"
_user_specified_name
states/0:QM
'
_output_shapes
:���������@
"
_user_specified_name
states/1
�K
�	
lstm_1_while_body_1394060*
&lstm_1_while_lstm_1_while_loop_counter0
,lstm_1_while_lstm_1_while_maximum_iterations
lstm_1_while_placeholder
lstm_1_while_placeholder_1
lstm_1_while_placeholder_2
lstm_1_while_placeholder_3)
%lstm_1_while_lstm_1_strided_slice_1_0e
alstm_1_while_tensorarrayv2read_tensorlistgetitem_lstm_1_tensorarrayunstack_tensorlistfromtensor_0=
9lstm_1_while_lstm_cell_2_matmul_readvariableop_resource_0?
;lstm_1_while_lstm_cell_2_matmul_1_readvariableop_resource_0>
:lstm_1_while_lstm_cell_2_biasadd_readvariableop_resource_0
lstm_1_while_identity
lstm_1_while_identity_1
lstm_1_while_identity_2
lstm_1_while_identity_3
lstm_1_while_identity_4
lstm_1_while_identity_5'
#lstm_1_while_lstm_1_strided_slice_1c
_lstm_1_while_tensorarrayv2read_tensorlistgetitem_lstm_1_tensorarrayunstack_tensorlistfromtensor;
7lstm_1_while_lstm_cell_2_matmul_readvariableop_resource=
9lstm_1_while_lstm_cell_2_matmul_1_readvariableop_resource<
8lstm_1_while_lstm_cell_2_biasadd_readvariableop_resource��/lstm_1/while/lstm_cell_2/BiasAdd/ReadVariableOp�.lstm_1/while/lstm_cell_2/MatMul/ReadVariableOp�0lstm_1/while/lstm_cell_2/MatMul_1/ReadVariableOp�
>lstm_1/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"2   @   2@
>lstm_1/while/TensorArrayV2Read/TensorListGetItem/element_shape�
0lstm_1/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemalstm_1_while_tensorarrayv2read_tensorlistgetitem_lstm_1_tensorarrayunstack_tensorlistfromtensor_0lstm_1_while_placeholderGlstm_1/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
_output_shapes

:2@*
element_dtype022
0lstm_1/while/TensorArrayV2Read/TensorListGetItem�
.lstm_1/while/lstm_cell_2/MatMul/ReadVariableOpReadVariableOp9lstm_1_while_lstm_cell_2_matmul_readvariableop_resource_0*
_output_shapes
:	@�*
dtype020
.lstm_1/while/lstm_cell_2/MatMul/ReadVariableOp�
lstm_1/while/lstm_cell_2/MatMulMatMul7lstm_1/while/TensorArrayV2Read/TensorListGetItem:item:06lstm_1/while/lstm_cell_2/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2�2!
lstm_1/while/lstm_cell_2/MatMul�
0lstm_1/while/lstm_cell_2/MatMul_1/ReadVariableOpReadVariableOp;lstm_1_while_lstm_cell_2_matmul_1_readvariableop_resource_0* 
_output_shapes
:
��*
dtype022
0lstm_1/while/lstm_cell_2/MatMul_1/ReadVariableOp�
!lstm_1/while/lstm_cell_2/MatMul_1MatMullstm_1_while_placeholder_28lstm_1/while/lstm_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	2�2#
!lstm_1/while/lstm_cell_2/MatMul_1�
lstm_1/while/lstm_cell_2/addAddV2)lstm_1/while/lstm_cell_2/MatMul:product:0+lstm_1/while/lstm_cell_2/MatMul_1:product:0*
T0*
_output_shapes
:	2�2
lstm_1/while/lstm_cell_2/add�
/lstm_1/while/lstm_cell_2/BiasAdd/ReadVariableOpReadVariableOp:lstm_1_while_lstm_cell_2_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype021
/lstm_1/while/lstm_cell_2/BiasAdd/ReadVariableOp�
 lstm_1/while/lstm_cell_2/BiasAddBiasAdd lstm_1/while/lstm_cell_2/add:z:07lstm_1/while/lstm_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	2�2"
 lstm_1/while/lstm_cell_2/BiasAdd�
lstm_1/while/lstm_cell_2/ConstConst*
_output_shapes
: *
dtype0*
value	B :2 
lstm_1/while/lstm_cell_2/Const�
(lstm_1/while/lstm_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2*
(lstm_1/while/lstm_cell_2/split/split_dim�
lstm_1/while/lstm_cell_2/splitSplit1lstm_1/while/lstm_cell_2/split/split_dim:output:0)lstm_1/while/lstm_cell_2/BiasAdd:output:0*
T0*@
_output_shapes.
,:	2�:	2�:	2�:	2�*
	num_split2 
lstm_1/while/lstm_cell_2/split�
 lstm_1/while/lstm_cell_2/SigmoidSigmoid'lstm_1/while/lstm_cell_2/split:output:0*
T0*
_output_shapes
:	2�2"
 lstm_1/while/lstm_cell_2/Sigmoid�
"lstm_1/while/lstm_cell_2/Sigmoid_1Sigmoid'lstm_1/while/lstm_cell_2/split:output:1*
T0*
_output_shapes
:	2�2$
"lstm_1/while/lstm_cell_2/Sigmoid_1�
lstm_1/while/lstm_cell_2/mulMul&lstm_1/while/lstm_cell_2/Sigmoid_1:y:0lstm_1_while_placeholder_3*
T0*
_output_shapes
:	2�2
lstm_1/while/lstm_cell_2/mul�
lstm_1/while/lstm_cell_2/ReluRelu'lstm_1/while/lstm_cell_2/split:output:2*
T0*
_output_shapes
:	2�2
lstm_1/while/lstm_cell_2/Relu�
lstm_1/while/lstm_cell_2/mul_1Mul$lstm_1/while/lstm_cell_2/Sigmoid:y:0+lstm_1/while/lstm_cell_2/Relu:activations:0*
T0*
_output_shapes
:	2�2 
lstm_1/while/lstm_cell_2/mul_1�
lstm_1/while/lstm_cell_2/add_1AddV2 lstm_1/while/lstm_cell_2/mul:z:0"lstm_1/while/lstm_cell_2/mul_1:z:0*
T0*
_output_shapes
:	2�2 
lstm_1/while/lstm_cell_2/add_1�
"lstm_1/while/lstm_cell_2/Sigmoid_2Sigmoid'lstm_1/while/lstm_cell_2/split:output:3*
T0*
_output_shapes
:	2�2$
"lstm_1/while/lstm_cell_2/Sigmoid_2�
lstm_1/while/lstm_cell_2/Relu_1Relu"lstm_1/while/lstm_cell_2/add_1:z:0*
T0*
_output_shapes
:	2�2!
lstm_1/while/lstm_cell_2/Relu_1�
lstm_1/while/lstm_cell_2/mul_2Mul&lstm_1/while/lstm_cell_2/Sigmoid_2:y:0-lstm_1/while/lstm_cell_2/Relu_1:activations:0*
T0*
_output_shapes
:	2�2 
lstm_1/while/lstm_cell_2/mul_2�
1lstm_1/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_1_while_placeholder_1lstm_1_while_placeholder"lstm_1/while/lstm_cell_2/mul_2:z:0*
_output_shapes
: *
element_dtype023
1lstm_1/while/TensorArrayV2Write/TensorListSetItemj
lstm_1/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_1/while/add/y�
lstm_1/while/addAddV2lstm_1_while_placeholderlstm_1/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_1/while/addn
lstm_1/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_1/while/add_1/y�
lstm_1/while/add_1AddV2&lstm_1_while_lstm_1_while_loop_counterlstm_1/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_1/while/add_1�
lstm_1/while/IdentityIdentitylstm_1/while/add_1:z:00^lstm_1/while/lstm_cell_2/BiasAdd/ReadVariableOp/^lstm_1/while/lstm_cell_2/MatMul/ReadVariableOp1^lstm_1/while/lstm_cell_2/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_1/while/Identity�
lstm_1/while/Identity_1Identity,lstm_1_while_lstm_1_while_maximum_iterations0^lstm_1/while/lstm_cell_2/BiasAdd/ReadVariableOp/^lstm_1/while/lstm_cell_2/MatMul/ReadVariableOp1^lstm_1/while/lstm_cell_2/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_1/while/Identity_1�
lstm_1/while/Identity_2Identitylstm_1/while/add:z:00^lstm_1/while/lstm_cell_2/BiasAdd/ReadVariableOp/^lstm_1/while/lstm_cell_2/MatMul/ReadVariableOp1^lstm_1/while/lstm_cell_2/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_1/while/Identity_2�
lstm_1/while/Identity_3IdentityAlstm_1/while/TensorArrayV2Write/TensorListSetItem:output_handle:00^lstm_1/while/lstm_cell_2/BiasAdd/ReadVariableOp/^lstm_1/while/lstm_cell_2/MatMul/ReadVariableOp1^lstm_1/while/lstm_cell_2/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_1/while/Identity_3�
lstm_1/while/Identity_4Identity"lstm_1/while/lstm_cell_2/mul_2:z:00^lstm_1/while/lstm_cell_2/BiasAdd/ReadVariableOp/^lstm_1/while/lstm_cell_2/MatMul/ReadVariableOp1^lstm_1/while/lstm_cell_2/MatMul_1/ReadVariableOp*
T0*
_output_shapes
:	2�2
lstm_1/while/Identity_4�
lstm_1/while/Identity_5Identity"lstm_1/while/lstm_cell_2/add_1:z:00^lstm_1/while/lstm_cell_2/BiasAdd/ReadVariableOp/^lstm_1/while/lstm_cell_2/MatMul/ReadVariableOp1^lstm_1/while/lstm_cell_2/MatMul_1/ReadVariableOp*
T0*
_output_shapes
:	2�2
lstm_1/while/Identity_5"7
lstm_1_while_identitylstm_1/while/Identity:output:0";
lstm_1_while_identity_1 lstm_1/while/Identity_1:output:0";
lstm_1_while_identity_2 lstm_1/while/Identity_2:output:0";
lstm_1_while_identity_3 lstm_1/while/Identity_3:output:0";
lstm_1_while_identity_4 lstm_1/while/Identity_4:output:0";
lstm_1_while_identity_5 lstm_1/while/Identity_5:output:0"L
#lstm_1_while_lstm_1_strided_slice_1%lstm_1_while_lstm_1_strided_slice_1_0"v
8lstm_1_while_lstm_cell_2_biasadd_readvariableop_resource:lstm_1_while_lstm_cell_2_biasadd_readvariableop_resource_0"x
9lstm_1_while_lstm_cell_2_matmul_1_readvariableop_resource;lstm_1_while_lstm_cell_2_matmul_1_readvariableop_resource_0"t
7lstm_1_while_lstm_cell_2_matmul_readvariableop_resource9lstm_1_while_lstm_cell_2_matmul_readvariableop_resource_0"�
_lstm_1_while_tensorarrayv2read_tensorlistgetitem_lstm_1_tensorarrayunstack_tensorlistfromtensoralstm_1_while_tensorarrayv2read_tensorlistgetitem_lstm_1_tensorarrayunstack_tensorlistfromtensor_0*A
_input_shapes0
.: : : : :	2�:	2�: : :::2b
/lstm_1/while/lstm_cell_2/BiasAdd/ReadVariableOp/lstm_1/while/lstm_cell_2/BiasAdd/ReadVariableOp2`
.lstm_1/while/lstm_cell_2/MatMul/ReadVariableOp.lstm_1/while/lstm_cell_2/MatMul/ReadVariableOp2d
0lstm_1/while/lstm_cell_2/MatMul_1/ReadVariableOp0lstm_1/while/lstm_cell_2/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	2�:%!

_output_shapes
:	2�:

_output_shapes
: :

_output_shapes
: 
�B
�
while_body_1395671
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_06
2while_lstm_cell_3_matmul_readvariableop_resource_08
4while_lstm_cell_3_matmul_1_readvariableop_resource_07
3while_lstm_cell_3_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor4
0while_lstm_cell_3_matmul_readvariableop_resource6
2while_lstm_cell_3_matmul_1_readvariableop_resource5
1while_lstm_cell_3_biasadd_readvariableop_resource��(while/lstm_cell_3/BiasAdd/ReadVariableOp�'while/lstm_cell_3/MatMul/ReadVariableOp�)while/lstm_cell_3/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�����   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape�
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:����������*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem�
'while/lstm_cell_3/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_3_matmul_readvariableop_resource_0* 
_output_shapes
:
��*
dtype02)
'while/lstm_cell_3/MatMul/ReadVariableOp�
while/lstm_cell_3/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_3/MatMul�
)while/lstm_cell_3/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_3_matmul_1_readvariableop_resource_0*
_output_shapes
:	@�*
dtype02+
)while/lstm_cell_3/MatMul_1/ReadVariableOp�
while/lstm_cell_3/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_3/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_3/MatMul_1�
while/lstm_cell_3/addAddV2"while/lstm_cell_3/MatMul:product:0$while/lstm_cell_3/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_3/add�
(while/lstm_cell_3/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_3_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02*
(while/lstm_cell_3/BiasAdd/ReadVariableOp�
while/lstm_cell_3/BiasAddBiasAddwhile/lstm_cell_3/add:z:00while/lstm_cell_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_3/BiasAddt
while/lstm_cell_3/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_3/Const�
!while/lstm_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_3/split/split_dim�
while/lstm_cell_3/splitSplit*while/lstm_cell_3/split/split_dim:output:0"while/lstm_cell_3/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������@:���������@:���������@:���������@*
	num_split2
while/lstm_cell_3/split�
while/lstm_cell_3/SigmoidSigmoid while/lstm_cell_3/split:output:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_3/Sigmoid�
while/lstm_cell_3/Sigmoid_1Sigmoid while/lstm_cell_3/split:output:1*
T0*'
_output_shapes
:���������@2
while/lstm_cell_3/Sigmoid_1�
while/lstm_cell_3/mulMulwhile/lstm_cell_3/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������@2
while/lstm_cell_3/mul�
while/lstm_cell_3/ReluRelu while/lstm_cell_3/split:output:2*
T0*'
_output_shapes
:���������@2
while/lstm_cell_3/Relu�
while/lstm_cell_3/mul_1Mulwhile/lstm_cell_3/Sigmoid:y:0$while/lstm_cell_3/Relu:activations:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_3/mul_1�
while/lstm_cell_3/add_1AddV2while/lstm_cell_3/mul:z:0while/lstm_cell_3/mul_1:z:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_3/add_1�
while/lstm_cell_3/Sigmoid_2Sigmoid while/lstm_cell_3/split:output:3*
T0*'
_output_shapes
:���������@2
while/lstm_cell_3/Sigmoid_2�
while/lstm_cell_3/Relu_1Reluwhile/lstm_cell_3/add_1:z:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_3/Relu_1�
while/lstm_cell_3/mul_2Mulwhile/lstm_cell_3/Sigmoid_2:y:0&while/lstm_cell_3/Relu_1:activations:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_3/mul_2�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_3/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1�
while/IdentityIdentitywhile/add_1:z:0)^while/lstm_cell_3/BiasAdd/ReadVariableOp(^while/lstm_cell_3/MatMul/ReadVariableOp*^while/lstm_cell_3/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity�
while/Identity_1Identitywhile_while_maximum_iterations)^while/lstm_cell_3/BiasAdd/ReadVariableOp(^while/lstm_cell_3/MatMul/ReadVariableOp*^while/lstm_cell_3/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1�
while/Identity_2Identitywhile/add:z:0)^while/lstm_cell_3/BiasAdd/ReadVariableOp(^while/lstm_cell_3/MatMul/ReadVariableOp*^while/lstm_cell_3/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2�
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0)^while/lstm_cell_3/BiasAdd/ReadVariableOp(^while/lstm_cell_3/MatMul/ReadVariableOp*^while/lstm_cell_3/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3�
while/Identity_4Identitywhile/lstm_cell_3/mul_2:z:0)^while/lstm_cell_3/BiasAdd/ReadVariableOp(^while/lstm_cell_3/MatMul/ReadVariableOp*^while/lstm_cell_3/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:���������@2
while/Identity_4�
while/Identity_5Identitywhile/lstm_cell_3/add_1:z:0)^while/lstm_cell_3/BiasAdd/ReadVariableOp(^while/lstm_cell_3/MatMul/ReadVariableOp*^while/lstm_cell_3/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:���������@2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_3_biasadd_readvariableop_resource3while_lstm_cell_3_biasadd_readvariableop_resource_0"j
2while_lstm_cell_3_matmul_1_readvariableop_resource4while_lstm_cell_3_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_3_matmul_readvariableop_resource2while_lstm_cell_3_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :���������@:���������@: : :::2T
(while/lstm_cell_3/BiasAdd/ReadVariableOp(while/lstm_cell_3/BiasAdd/ReadVariableOp2R
'while/lstm_cell_3/MatMul/ReadVariableOp'while/lstm_cell_3/MatMul/ReadVariableOp2V
)while/lstm_cell_3/MatMul_1/ReadVariableOp)while/lstm_cell_3/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������@:-)
'
_output_shapes
:���������@:

_output_shapes
: :

_output_shapes
: 
�
�
H__inference_lstm_cell_2_layer_call_and_return_conditional_losses_1396044

inputs
states_0
states_1"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1

identity_2��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@�*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
��*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:����������2
add�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:����������:����������:����������:����������*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:����������2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:����������2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:����������2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:����������2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:����������2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:����������2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:����������2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:����������2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:����������2
mul_2�
IdentityIdentity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity�

Identity_1Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity_1�

Identity_2Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*Z
_input_shapesI
G:���������@:����������:����������:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs:RN
(
_output_shapes
:����������
"
_user_specified_name
states/0:RN
(
_output_shapes
:����������
"
_user_specified_name
states/1
�
�
while_cond_1394533
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1394533___redundant_placeholder05
1while_while_cond_1394533___redundant_placeholder15
1while_while_cond_1394533___redundant_placeholder25
1while_while_cond_1394533___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :���������@:���������@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������@:-)
'
_output_shapes
:���������@:

_output_shapes
: :

_output_shapes
:
�
�
H__inference_lstm_cell_1_layer_call_and_return_conditional_losses_1392084

inputs

states
states_1"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1

identity_2��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	\�*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	@�*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:����������2
add�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:���������@:���������@:���������@:���������@*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:���������@2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:���������@2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:���������@2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:���������@2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:���������@2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:���������@2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:���������@2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:���������@2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:���������@2
mul_2�
IdentityIdentity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:���������@2

Identity�

Identity_1Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:���������@2

Identity_1�

Identity_2Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:���������@2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:���������\:���������@:���������@:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:���������\
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������@
 
_user_specified_namestates:OK
'
_output_shapes
:���������@
 
_user_specified_namestates
�
�
(__inference_lstm_1_layer_call_fn_1395450
inputs_0
unknown
	unknown_0
	unknown_1
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:�������������������*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_lstm_1_layer_call_and_return_conditional_losses_13932222
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*5
_output_shapes#
!:�������������������2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:������������������@:::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :������������������@
"
_user_specified_name
inputs/0
�
�
H__inference_lstm_cell_2_layer_call_and_return_conditional_losses_1392694

inputs

states
states_1"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1

identity_2��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@�*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
��*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:����������2
add�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:����������:����������:����������:����������*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:����������2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:����������2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:����������2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:����������2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:����������2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:����������2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:����������2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:����������2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:����������2
mul_2�
IdentityIdentity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity�

Identity_1Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity_1�

Identity_2Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*Z
_input_shapesI
G:���������@:����������:����������:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs:PL
(
_output_shapes
:����������
 
_user_specified_namestates:PL
(
_output_shapes
:����������
 
_user_specified_namestates
�[
�
C__inference_lstm_1_layer_call_and_return_conditional_losses_1395275
inputs_0.
*lstm_cell_2_matmul_readvariableop_resource0
,lstm_cell_2_matmul_1_readvariableop_resource/
+lstm_cell_2_biasadd_readvariableop_resource
identity��"lstm_cell_2/BiasAdd/ReadVariableOp�!lstm_cell_2/MatMul/ReadVariableOp�#lstm_cell_2/MatMul_1/ReadVariableOp�whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :�2
zeros/packed/1�
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:����������2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :�2
zeros_1/packed/1�
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:����������2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm�
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :������������������@2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1�
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2
TensorArrayV2/element_shape�
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2�
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   27
5TensorArrayUnstack/TensorListFromTensor/element_shape�
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������@*
shrink_axis_mask2
strided_slice_2�
!lstm_cell_2/MatMul/ReadVariableOpReadVariableOp*lstm_cell_2_matmul_readvariableop_resource*
_output_shapes
:	@�*
dtype02#
!lstm_cell_2/MatMul/ReadVariableOp�
lstm_cell_2/MatMulMatMulstrided_slice_2:output:0)lstm_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_2/MatMul�
#lstm_cell_2/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_2_matmul_1_readvariableop_resource* 
_output_shapes
:
��*
dtype02%
#lstm_cell_2/MatMul_1/ReadVariableOp�
lstm_cell_2/MatMul_1MatMulzeros:output:0+lstm_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_2/MatMul_1�
lstm_cell_2/addAddV2lstm_cell_2/MatMul:product:0lstm_cell_2/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
lstm_cell_2/add�
"lstm_cell_2/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02$
"lstm_cell_2/BiasAdd/ReadVariableOp�
lstm_cell_2/BiasAddBiasAddlstm_cell_2/add:z:0*lstm_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_2/BiasAddh
lstm_cell_2/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_2/Const|
lstm_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_2/split/split_dim�
lstm_cell_2/splitSplit$lstm_cell_2/split/split_dim:output:0lstm_cell_2/BiasAdd:output:0*
T0*d
_output_shapesR
P:����������:����������:����������:����������*
	num_split2
lstm_cell_2/split�
lstm_cell_2/SigmoidSigmoidlstm_cell_2/split:output:0*
T0*(
_output_shapes
:����������2
lstm_cell_2/Sigmoid�
lstm_cell_2/Sigmoid_1Sigmoidlstm_cell_2/split:output:1*
T0*(
_output_shapes
:����������2
lstm_cell_2/Sigmoid_1�
lstm_cell_2/mulMullstm_cell_2/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:����������2
lstm_cell_2/mul{
lstm_cell_2/ReluRelulstm_cell_2/split:output:2*
T0*(
_output_shapes
:����������2
lstm_cell_2/Relu�
lstm_cell_2/mul_1Mullstm_cell_2/Sigmoid:y:0lstm_cell_2/Relu:activations:0*
T0*(
_output_shapes
:����������2
lstm_cell_2/mul_1�
lstm_cell_2/add_1AddV2lstm_cell_2/mul:z:0lstm_cell_2/mul_1:z:0*
T0*(
_output_shapes
:����������2
lstm_cell_2/add_1�
lstm_cell_2/Sigmoid_2Sigmoidlstm_cell_2/split:output:3*
T0*(
_output_shapes
:����������2
lstm_cell_2/Sigmoid_2z
lstm_cell_2/Relu_1Relulstm_cell_2/add_1:z:0*
T0*(
_output_shapes
:����������2
lstm_cell_2/Relu_1�
lstm_cell_2/mul_2Mullstm_cell_2/Sigmoid_2:y:0 lstm_cell_2/Relu_1:activations:0*
T0*(
_output_shapes
:����������2
lstm_cell_2/mul_2�
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�����   2
TensorArrayV2_1/element_shape�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter�
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_2_matmul_readvariableop_resource,lstm_cell_2_matmul_1_readvariableop_resource+lstm_cell_2_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :����������:����������: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_1395190*
condR
while_cond_1395189*M
output_shapes<
:: : : : :����������:����������: : : : : *
parallel_iterations 2
while�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�����   22
0TensorArrayV2Stack/TensorListStack/element_shape�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:�������������������*
element_dtype02$
"TensorArrayV2Stack/TensorListStack�
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:�������������������2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime�
IdentityIdentitytranspose_1:y:0#^lstm_cell_2/BiasAdd/ReadVariableOp"^lstm_cell_2/MatMul/ReadVariableOp$^lstm_cell_2/MatMul_1/ReadVariableOp^while*
T0*5
_output_shapes#
!:�������������������2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:������������������@:::2H
"lstm_cell_2/BiasAdd/ReadVariableOp"lstm_cell_2/BiasAdd/ReadVariableOp2F
!lstm_cell_2/MatMul/ReadVariableOp!lstm_cell_2/MatMul/ReadVariableOp2J
#lstm_cell_2/MatMul_1/ReadVariableOp#lstm_cell_2/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :������������������@
"
_user_specified_name
inputs/0
�
�
-__inference_lstm_cell_2_layer_call_fn_1396061

inputs
states_0
states_1
unknown
	unknown_0
	unknown_1
identity

identity_1

identity_2��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:����������:����������:����������*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_lstm_cell_2_layer_call_and_return_conditional_losses_13926942
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity�

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity_1�

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*Z
_input_shapesI
G:���������@:����������:����������:::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs:RN
(
_output_shapes
:����������
"
_user_specified_name
states/0:RN
(
_output_shapes
:����������
"
_user_specified_name
states/1
�
�
while_cond_1395014
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1395014___redundant_placeholder05
1while_while_cond_1395014___redundant_placeholder15
1while_while_cond_1395014___redundant_placeholder25
1while_while_cond_1395014___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :���������@:���������@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������@:-)
'
_output_shapes
:���������@:

_output_shapes
: :

_output_shapes
:
�%
�
while_body_1392411
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_lstm_cell_1_1392435_0
while_lstm_cell_1_1392437_0
while_lstm_cell_1_1392439_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_lstm_cell_1_1392435
while_lstm_cell_1_1392437
while_lstm_cell_1_1392439��)while/lstm_cell_1/StatefulPartitionedCall�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����\   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape�
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������\*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem�
)while/lstm_cell_1/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_1_1392435_0while_lstm_cell_1_1392437_0while_lstm_cell_1_1392439_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������@:���������@:���������@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_lstm_cell_1_layer_call_and_return_conditional_losses_13920842+
)while/lstm_cell_1/StatefulPartitionedCall�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/lstm_cell_1/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1�
while/IdentityIdentitywhile/add_1:z:0*^while/lstm_cell_1/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity�
while/Identity_1Identitywhile_while_maximum_iterations*^while/lstm_cell_1/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1�
while/Identity_2Identitywhile/add:z:0*^while/lstm_cell_1/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2�
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*^while/lstm_cell_1/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3�
while/Identity_4Identity2while/lstm_cell_1/StatefulPartitionedCall:output:1*^while/lstm_cell_1/StatefulPartitionedCall*
T0*'
_output_shapes
:���������@2
while/Identity_4�
while/Identity_5Identity2while/lstm_cell_1/StatefulPartitionedCall:output:2*^while/lstm_cell_1/StatefulPartitionedCall*
T0*'
_output_shapes
:���������@2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"8
while_lstm_cell_1_1392435while_lstm_cell_1_1392435_0"8
while_lstm_cell_1_1392437while_lstm_cell_1_1392437_0"8
while_lstm_cell_1_1392439while_lstm_cell_1_1392439_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :���������@:���������@: : :::2V
)while/lstm_cell_1/StatefulPartitionedCall)while/lstm_cell_1/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������@:-)
'
_output_shapes
:���������@:

_output_shapes
: :

_output_shapes
: 
�H
�	
lstm_while_body_1393911&
"lstm_while_lstm_while_loop_counter,
(lstm_while_lstm_while_maximum_iterations
lstm_while_placeholder
lstm_while_placeholder_1
lstm_while_placeholder_2
lstm_while_placeholder_3%
!lstm_while_lstm_strided_slice_1_0a
]lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor_0;
7lstm_while_lstm_cell_1_matmul_readvariableop_resource_0=
9lstm_while_lstm_cell_1_matmul_1_readvariableop_resource_0<
8lstm_while_lstm_cell_1_biasadd_readvariableop_resource_0
lstm_while_identity
lstm_while_identity_1
lstm_while_identity_2
lstm_while_identity_3
lstm_while_identity_4
lstm_while_identity_5#
lstm_while_lstm_strided_slice_1_
[lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor9
5lstm_while_lstm_cell_1_matmul_readvariableop_resource;
7lstm_while_lstm_cell_1_matmul_1_readvariableop_resource:
6lstm_while_lstm_cell_1_biasadd_readvariableop_resource��-lstm/while/lstm_cell_1/BiasAdd/ReadVariableOp�,lstm/while/lstm_cell_1/MatMul/ReadVariableOp�.lstm/while/lstm_cell_1/MatMul_1/ReadVariableOp�
<lstm/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"2   \   2>
<lstm/while/TensorArrayV2Read/TensorListGetItem/element_shape�
.lstm/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem]lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor_0lstm_while_placeholderElstm/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
_output_shapes

:2\*
element_dtype020
.lstm/while/TensorArrayV2Read/TensorListGetItem�
,lstm/while/lstm_cell_1/MatMul/ReadVariableOpReadVariableOp7lstm_while_lstm_cell_1_matmul_readvariableop_resource_0*
_output_shapes
:	\�*
dtype02.
,lstm/while/lstm_cell_1/MatMul/ReadVariableOp�
lstm/while/lstm_cell_1/MatMulMatMul5lstm/while/TensorArrayV2Read/TensorListGetItem:item:04lstm/while/lstm_cell_1/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2�2
lstm/while/lstm_cell_1/MatMul�
.lstm/while/lstm_cell_1/MatMul_1/ReadVariableOpReadVariableOp9lstm_while_lstm_cell_1_matmul_1_readvariableop_resource_0*
_output_shapes
:	@�*
dtype020
.lstm/while/lstm_cell_1/MatMul_1/ReadVariableOp�
lstm/while/lstm_cell_1/MatMul_1MatMullstm_while_placeholder_26lstm/while/lstm_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	2�2!
lstm/while/lstm_cell_1/MatMul_1�
lstm/while/lstm_cell_1/addAddV2'lstm/while/lstm_cell_1/MatMul:product:0)lstm/while/lstm_cell_1/MatMul_1:product:0*
T0*
_output_shapes
:	2�2
lstm/while/lstm_cell_1/add�
-lstm/while/lstm_cell_1/BiasAdd/ReadVariableOpReadVariableOp8lstm_while_lstm_cell_1_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02/
-lstm/while/lstm_cell_1/BiasAdd/ReadVariableOp�
lstm/while/lstm_cell_1/BiasAddBiasAddlstm/while/lstm_cell_1/add:z:05lstm/while/lstm_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	2�2 
lstm/while/lstm_cell_1/BiasAdd~
lstm/while/lstm_cell_1/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm/while/lstm_cell_1/Const�
&lstm/while/lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2(
&lstm/while/lstm_cell_1/split/split_dim�
lstm/while/lstm_cell_1/splitSplit/lstm/while/lstm_cell_1/split/split_dim:output:0'lstm/while/lstm_cell_1/BiasAdd:output:0*
T0*<
_output_shapes*
(:2@:2@:2@:2@*
	num_split2
lstm/while/lstm_cell_1/split�
lstm/while/lstm_cell_1/SigmoidSigmoid%lstm/while/lstm_cell_1/split:output:0*
T0*
_output_shapes

:2@2 
lstm/while/lstm_cell_1/Sigmoid�
 lstm/while/lstm_cell_1/Sigmoid_1Sigmoid%lstm/while/lstm_cell_1/split:output:1*
T0*
_output_shapes

:2@2"
 lstm/while/lstm_cell_1/Sigmoid_1�
lstm/while/lstm_cell_1/mulMul$lstm/while/lstm_cell_1/Sigmoid_1:y:0lstm_while_placeholder_3*
T0*
_output_shapes

:2@2
lstm/while/lstm_cell_1/mul�
lstm/while/lstm_cell_1/ReluRelu%lstm/while/lstm_cell_1/split:output:2*
T0*
_output_shapes

:2@2
lstm/while/lstm_cell_1/Relu�
lstm/while/lstm_cell_1/mul_1Mul"lstm/while/lstm_cell_1/Sigmoid:y:0)lstm/while/lstm_cell_1/Relu:activations:0*
T0*
_output_shapes

:2@2
lstm/while/lstm_cell_1/mul_1�
lstm/while/lstm_cell_1/add_1AddV2lstm/while/lstm_cell_1/mul:z:0 lstm/while/lstm_cell_1/mul_1:z:0*
T0*
_output_shapes

:2@2
lstm/while/lstm_cell_1/add_1�
 lstm/while/lstm_cell_1/Sigmoid_2Sigmoid%lstm/while/lstm_cell_1/split:output:3*
T0*
_output_shapes

:2@2"
 lstm/while/lstm_cell_1/Sigmoid_2�
lstm/while/lstm_cell_1/Relu_1Relu lstm/while/lstm_cell_1/add_1:z:0*
T0*
_output_shapes

:2@2
lstm/while/lstm_cell_1/Relu_1�
lstm/while/lstm_cell_1/mul_2Mul$lstm/while/lstm_cell_1/Sigmoid_2:y:0+lstm/while/lstm_cell_1/Relu_1:activations:0*
T0*
_output_shapes

:2@2
lstm/while/lstm_cell_1/mul_2�
/lstm/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_while_placeholder_1lstm_while_placeholder lstm/while/lstm_cell_1/mul_2:z:0*
_output_shapes
: *
element_dtype021
/lstm/while/TensorArrayV2Write/TensorListSetItemf
lstm/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm/while/add/y}
lstm/while/addAddV2lstm_while_placeholderlstm/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm/while/addj
lstm/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm/while/add_1/y�
lstm/while/add_1AddV2"lstm_while_lstm_while_loop_counterlstm/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm/while/add_1�
lstm/while/IdentityIdentitylstm/while/add_1:z:0.^lstm/while/lstm_cell_1/BiasAdd/ReadVariableOp-^lstm/while/lstm_cell_1/MatMul/ReadVariableOp/^lstm/while/lstm_cell_1/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm/while/Identity�
lstm/while/Identity_1Identity(lstm_while_lstm_while_maximum_iterations.^lstm/while/lstm_cell_1/BiasAdd/ReadVariableOp-^lstm/while/lstm_cell_1/MatMul/ReadVariableOp/^lstm/while/lstm_cell_1/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm/while/Identity_1�
lstm/while/Identity_2Identitylstm/while/add:z:0.^lstm/while/lstm_cell_1/BiasAdd/ReadVariableOp-^lstm/while/lstm_cell_1/MatMul/ReadVariableOp/^lstm/while/lstm_cell_1/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm/while/Identity_2�
lstm/while/Identity_3Identity?lstm/while/TensorArrayV2Write/TensorListSetItem:output_handle:0.^lstm/while/lstm_cell_1/BiasAdd/ReadVariableOp-^lstm/while/lstm_cell_1/MatMul/ReadVariableOp/^lstm/while/lstm_cell_1/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm/while/Identity_3�
lstm/while/Identity_4Identity lstm/while/lstm_cell_1/mul_2:z:0.^lstm/while/lstm_cell_1/BiasAdd/ReadVariableOp-^lstm/while/lstm_cell_1/MatMul/ReadVariableOp/^lstm/while/lstm_cell_1/MatMul_1/ReadVariableOp*
T0*
_output_shapes

:2@2
lstm/while/Identity_4�
lstm/while/Identity_5Identity lstm/while/lstm_cell_1/add_1:z:0.^lstm/while/lstm_cell_1/BiasAdd/ReadVariableOp-^lstm/while/lstm_cell_1/MatMul/ReadVariableOp/^lstm/while/lstm_cell_1/MatMul_1/ReadVariableOp*
T0*
_output_shapes

:2@2
lstm/while/Identity_5"3
lstm_while_identitylstm/while/Identity:output:0"7
lstm_while_identity_1lstm/while/Identity_1:output:0"7
lstm_while_identity_2lstm/while/Identity_2:output:0"7
lstm_while_identity_3lstm/while/Identity_3:output:0"7
lstm_while_identity_4lstm/while/Identity_4:output:0"7
lstm_while_identity_5lstm/while/Identity_5:output:0"r
6lstm_while_lstm_cell_1_biasadd_readvariableop_resource8lstm_while_lstm_cell_1_biasadd_readvariableop_resource_0"t
7lstm_while_lstm_cell_1_matmul_1_readvariableop_resource9lstm_while_lstm_cell_1_matmul_1_readvariableop_resource_0"p
5lstm_while_lstm_cell_1_matmul_readvariableop_resource7lstm_while_lstm_cell_1_matmul_readvariableop_resource_0"D
lstm_while_lstm_strided_slice_1!lstm_while_lstm_strided_slice_1_0"�
[lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor]lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor_0*?
_input_shapes.
,: : : : :2@:2@: : :::2^
-lstm/while/lstm_cell_1/BiasAdd/ReadVariableOp-lstm/while/lstm_cell_1/BiasAdd/ReadVariableOp2\
,lstm/while/lstm_cell_1/MatMul/ReadVariableOp,lstm/while/lstm_cell_1/MatMul/ReadVariableOp2`
.lstm/while/lstm_cell_1/MatMul_1/ReadVariableOp.lstm/while/lstm_cell_1/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:2@:$ 

_output_shapes

:2@:

_output_shapes
: :

_output_shapes
: 
�
�
while_cond_1393762
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1393762___redundant_placeholder05
1while_while_cond_1393762___redundant_placeholder15
1while_while_cond_1393762___redundant_placeholder25
1while_while_cond_1393762___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :���������@:���������@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������@:-)
'
_output_shapes
:���������@:

_output_shapes
: :

_output_shapes
:
�	
�
lstm_cc_while_cond_1394372,
(lstm_cc_while_lstm_cc_while_loop_counter2
.lstm_cc_while_lstm_cc_while_maximum_iterations
lstm_cc_while_placeholder
lstm_cc_while_placeholder_1
lstm_cc_while_placeholder_2
lstm_cc_while_placeholder_3.
*lstm_cc_while_less_lstm_cc_strided_slice_1E
Alstm_cc_while_lstm_cc_while_cond_1394372___redundant_placeholder0E
Alstm_cc_while_lstm_cc_while_cond_1394372___redundant_placeholder1E
Alstm_cc_while_lstm_cc_while_cond_1394372___redundant_placeholder2E
Alstm_cc_while_lstm_cc_while_cond_1394372___redundant_placeholder3
lstm_cc_while_identity
�
lstm_cc/while/LessLesslstm_cc_while_placeholder*lstm_cc_while_less_lstm_cc_strided_slice_1*
T0*
_output_shapes
: 2
lstm_cc/while/Lessu
lstm_cc/while/IdentityIdentitylstm_cc/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_cc/while/Identity"9
lstm_cc_while_identitylstm_cc/while/Identity:output:0*A
_input_shapes0
.: : : : :2@:2@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:2@:$ 

_output_shapes

:2@:

_output_shapes
: :

_output_shapes
:
�$
�
while_body_1391933
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_lstm_cell_1391957_0
while_lstm_cell_1391959_0
while_lstm_cell_1391961_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_lstm_cell_1391957
while_lstm_cell_1391959
while_lstm_cell_1391961��'while/lstm_cell/StatefulPartitionedCall�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����\   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape�
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������\*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem�
'while/lstm_cell/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_1391957_0while_lstm_cell_1391959_0while_lstm_cell_1391961_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������@:���������@:���������@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_lstm_cell_layer_call_and_return_conditional_losses_13915072)
'while/lstm_cell/StatefulPartitionedCall�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder0while/lstm_cell/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1�
while/IdentityIdentitywhile/add_1:z:0(^while/lstm_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity�
while/Identity_1Identitywhile_while_maximum_iterations(^while/lstm_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1�
while/Identity_2Identitywhile/add:z:0(^while/lstm_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2�
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/lstm_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3�
while/Identity_4Identity0while/lstm_cell/StatefulPartitionedCall:output:1(^while/lstm_cell/StatefulPartitionedCall*
T0*'
_output_shapes
:���������@2
while/Identity_4�
while/Identity_5Identity0while/lstm_cell/StatefulPartitionedCall:output:2(^while/lstm_cell/StatefulPartitionedCall*
T0*'
_output_shapes
:���������@2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"4
while_lstm_cell_1391957while_lstm_cell_1391957_0"4
while_lstm_cell_1391959while_lstm_cell_1391959_0"4
while_lstm_cell_1391961while_lstm_cell_1391961_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :���������@:���������@: : :::2R
'while/lstm_cell/StatefulPartitionedCall'while/lstm_cell/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������@:-)
'
_output_shapes
:���������@:

_output_shapes
: :

_output_shapes
: 
�
�
while_cond_1391800
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1391800___redundant_placeholder05
1while_while_cond_1391800___redundant_placeholder15
1while_while_cond_1391800___redundant_placeholder25
1while_while_cond_1391800___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :���������@:���������@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������@:-)
'
_output_shapes
:���������@:

_output_shapes
: :

_output_shapes
:
�%
�
while_body_1393021
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_lstm_cell_2_1393045_0
while_lstm_cell_2_1393047_0
while_lstm_cell_2_1393049_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_lstm_cell_2_1393045
while_lstm_cell_2_1393047
while_lstm_cell_2_1393049��)while/lstm_cell_2/StatefulPartitionedCall�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape�
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������@*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem�
)while/lstm_cell_2/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_2_1393045_0while_lstm_cell_2_1393047_0while_lstm_cell_2_1393049_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:����������:����������:����������*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_lstm_cell_2_layer_call_and_return_conditional_losses_13926942+
)while/lstm_cell_2/StatefulPartitionedCall�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/lstm_cell_2/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1�
while/IdentityIdentitywhile/add_1:z:0*^while/lstm_cell_2/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity�
while/Identity_1Identitywhile_while_maximum_iterations*^while/lstm_cell_2/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1�
while/Identity_2Identitywhile/add:z:0*^while/lstm_cell_2/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2�
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*^while/lstm_cell_2/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3�
while/Identity_4Identity2while/lstm_cell_2/StatefulPartitionedCall:output:1*^while/lstm_cell_2/StatefulPartitionedCall*
T0*(
_output_shapes
:����������2
while/Identity_4�
while/Identity_5Identity2while/lstm_cell_2/StatefulPartitionedCall:output:2*^while/lstm_cell_2/StatefulPartitionedCall*
T0*(
_output_shapes
:����������2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"8
while_lstm_cell_2_1393045while_lstm_cell_2_1393045_0"8
while_lstm_cell_2_1393047while_lstm_cell_2_1393047_0"8
while_lstm_cell_2_1393049while_lstm_cell_2_1393049_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :����������:����������: : :::2V
)while/lstm_cell_2/StatefulPartitionedCall)while/lstm_cell_2/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:����������:.*
(
_output_shapes
:����������:

_output_shapes
: :

_output_shapes
: 
�Z
�
A__inference_lstm_layer_call_and_return_conditional_losses_1394947
inputs_0.
*lstm_cell_1_matmul_readvariableop_resource0
,lstm_cell_1_matmul_1_readvariableop_resource/
+lstm_cell_1_biasadd_readvariableop_resource
identity��"lstm_cell_1/BiasAdd/ReadVariableOp�!lstm_cell_1/MatMul/ReadVariableOp�#lstm_cell_1/MatMul_1/ReadVariableOp�whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros/packed/1�
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������@2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/packed/1�
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:���������@2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm�
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :������������������\2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1�
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2
TensorArrayV2/element_shape�
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2�
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����\   27
5TensorArrayUnstack/TensorListFromTensor/element_shape�
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������\*
shrink_axis_mask2
strided_slice_2�
!lstm_cell_1/MatMul/ReadVariableOpReadVariableOp*lstm_cell_1_matmul_readvariableop_resource*
_output_shapes
:	\�*
dtype02#
!lstm_cell_1/MatMul/ReadVariableOp�
lstm_cell_1/MatMulMatMulstrided_slice_2:output:0)lstm_cell_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_1/MatMul�
#lstm_cell_1/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_1_matmul_1_readvariableop_resource*
_output_shapes
:	@�*
dtype02%
#lstm_cell_1/MatMul_1/ReadVariableOp�
lstm_cell_1/MatMul_1MatMulzeros:output:0+lstm_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_1/MatMul_1�
lstm_cell_1/addAddV2lstm_cell_1/MatMul:product:0lstm_cell_1/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
lstm_cell_1/add�
"lstm_cell_1/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02$
"lstm_cell_1/BiasAdd/ReadVariableOp�
lstm_cell_1/BiasAddBiasAddlstm_cell_1/add:z:0*lstm_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_1/BiasAddh
lstm_cell_1/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_1/Const|
lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_1/split/split_dim�
lstm_cell_1/splitSplit$lstm_cell_1/split/split_dim:output:0lstm_cell_1/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������@:���������@:���������@:���������@*
	num_split2
lstm_cell_1/split�
lstm_cell_1/SigmoidSigmoidlstm_cell_1/split:output:0*
T0*'
_output_shapes
:���������@2
lstm_cell_1/Sigmoid�
lstm_cell_1/Sigmoid_1Sigmoidlstm_cell_1/split:output:1*
T0*'
_output_shapes
:���������@2
lstm_cell_1/Sigmoid_1�
lstm_cell_1/mulMullstm_cell_1/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������@2
lstm_cell_1/mulz
lstm_cell_1/ReluRelulstm_cell_1/split:output:2*
T0*'
_output_shapes
:���������@2
lstm_cell_1/Relu�
lstm_cell_1/mul_1Mullstm_cell_1/Sigmoid:y:0lstm_cell_1/Relu:activations:0*
T0*'
_output_shapes
:���������@2
lstm_cell_1/mul_1�
lstm_cell_1/add_1AddV2lstm_cell_1/mul:z:0lstm_cell_1/mul_1:z:0*
T0*'
_output_shapes
:���������@2
lstm_cell_1/add_1�
lstm_cell_1/Sigmoid_2Sigmoidlstm_cell_1/split:output:3*
T0*'
_output_shapes
:���������@2
lstm_cell_1/Sigmoid_2y
lstm_cell_1/Relu_1Relulstm_cell_1/add_1:z:0*
T0*'
_output_shapes
:���������@2
lstm_cell_1/Relu_1�
lstm_cell_1/mul_2Mullstm_cell_1/Sigmoid_2:y:0 lstm_cell_1/Relu_1:activations:0*
T0*'
_output_shapes
:���������@2
lstm_cell_1/mul_2�
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   2
TensorArrayV2_1/element_shape�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter�
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_1_matmul_readvariableop_resource,lstm_cell_1_matmul_1_readvariableop_resource+lstm_cell_1_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������@:���������@: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_1394862*
condR
while_cond_1394861*K
output_shapes:
8: : : : :���������@:���������@: : : : : *
parallel_iterations 2
while�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   22
0TensorArrayV2Stack/TensorListStack/element_shape�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������@*
element_dtype02$
"TensorArrayV2Stack/TensorListStack�
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������@*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������@2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime�
IdentityIdentitytranspose_1:y:0#^lstm_cell_1/BiasAdd/ReadVariableOp"^lstm_cell_1/MatMul/ReadVariableOp$^lstm_cell_1/MatMul_1/ReadVariableOp^while*
T0*4
_output_shapes"
 :������������������@2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:������������������\:::2H
"lstm_cell_1/BiasAdd/ReadVariableOp"lstm_cell_1/BiasAdd/ReadVariableOp2F
!lstm_cell_1/MatMul/ReadVariableOp!lstm_cell_1/MatMul/ReadVariableOp2J
#lstm_cell_1/MatMul_1/ReadVariableOp#lstm_cell_1/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :������������������\
"
_user_specified_name
inputs/0
�
�
H__inference_lstm_cell_2_layer_call_and_return_conditional_losses_1392727

inputs

states
states_1"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1

identity_2��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@�*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
��*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:����������2
add�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:����������:����������:����������:����������*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:����������2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:����������2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:����������2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:����������2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:����������2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:����������2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:����������2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:����������2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:����������2
mul_2�
IdentityIdentity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity�

Identity_1Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity_1�

Identity_2Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*Z
_input_shapesI
G:���������@:����������:����������:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs:PL
(
_output_shapes
:����������
 
_user_specified_namestates:PL
(
_output_shapes
:����������
 
_user_specified_namestates
�
�
(__inference_lstm_2_layer_call_fn_1395767
inputs_0
unknown
	unknown_0
	unknown_1
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_lstm_2_layer_call_and_return_conditional_losses_13937002
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:�������������������:::22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:�������������������
"
_user_specified_name
inputs/0
�
�
-__inference_lstm_cell_3_layer_call_fn_1396178

inputs
states_0
states_1
unknown
	unknown_0
	unknown_1
identity

identity_1

identity_2��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������@:���������@:���������@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_lstm_cell_3_layer_call_and_return_conditional_losses_13933372
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������@2

Identity�

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:���������@2

Identity_1�

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:���������@2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*Y
_input_shapesH
F:����������:���������@:���������@:::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:QM
'
_output_shapes
:���������@
"
_user_specified_name
states/0:QM
'
_output_shapes
:���������@
"
_user_specified_name
states/1
�
�
while_cond_1395670
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1395670___redundant_placeholder05
1while_while_cond_1395670___redundant_placeholder15
1while_while_cond_1395670___redundant_placeholder25
1while_while_cond_1395670___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :���������@:���������@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������@:-)
'
_output_shapes
:���������@:

_output_shapes
: :

_output_shapes
:
�B
�
while_body_1395015
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_06
2while_lstm_cell_1_matmul_readvariableop_resource_08
4while_lstm_cell_1_matmul_1_readvariableop_resource_07
3while_lstm_cell_1_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor4
0while_lstm_cell_1_matmul_readvariableop_resource6
2while_lstm_cell_1_matmul_1_readvariableop_resource5
1while_lstm_cell_1_biasadd_readvariableop_resource��(while/lstm_cell_1/BiasAdd/ReadVariableOp�'while/lstm_cell_1/MatMul/ReadVariableOp�)while/lstm_cell_1/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����\   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape�
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������\*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem�
'while/lstm_cell_1/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_1_matmul_readvariableop_resource_0*
_output_shapes
:	\�*
dtype02)
'while/lstm_cell_1/MatMul/ReadVariableOp�
while/lstm_cell_1/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_1/MatMul�
)while/lstm_cell_1/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_1_matmul_1_readvariableop_resource_0*
_output_shapes
:	@�*
dtype02+
)while/lstm_cell_1/MatMul_1/ReadVariableOp�
while/lstm_cell_1/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_1/MatMul_1�
while/lstm_cell_1/addAddV2"while/lstm_cell_1/MatMul:product:0$while/lstm_cell_1/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_1/add�
(while/lstm_cell_1/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_1_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02*
(while/lstm_cell_1/BiasAdd/ReadVariableOp�
while/lstm_cell_1/BiasAddBiasAddwhile/lstm_cell_1/add:z:00while/lstm_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_1/BiasAddt
while/lstm_cell_1/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_1/Const�
!while/lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_1/split/split_dim�
while/lstm_cell_1/splitSplit*while/lstm_cell_1/split/split_dim:output:0"while/lstm_cell_1/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������@:���������@:���������@:���������@*
	num_split2
while/lstm_cell_1/split�
while/lstm_cell_1/SigmoidSigmoid while/lstm_cell_1/split:output:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_1/Sigmoid�
while/lstm_cell_1/Sigmoid_1Sigmoid while/lstm_cell_1/split:output:1*
T0*'
_output_shapes
:���������@2
while/lstm_cell_1/Sigmoid_1�
while/lstm_cell_1/mulMulwhile/lstm_cell_1/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������@2
while/lstm_cell_1/mul�
while/lstm_cell_1/ReluRelu while/lstm_cell_1/split:output:2*
T0*'
_output_shapes
:���������@2
while/lstm_cell_1/Relu�
while/lstm_cell_1/mul_1Mulwhile/lstm_cell_1/Sigmoid:y:0$while/lstm_cell_1/Relu:activations:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_1/mul_1�
while/lstm_cell_1/add_1AddV2while/lstm_cell_1/mul:z:0while/lstm_cell_1/mul_1:z:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_1/add_1�
while/lstm_cell_1/Sigmoid_2Sigmoid while/lstm_cell_1/split:output:3*
T0*'
_output_shapes
:���������@2
while/lstm_cell_1/Sigmoid_2�
while/lstm_cell_1/Relu_1Reluwhile/lstm_cell_1/add_1:z:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_1/Relu_1�
while/lstm_cell_1/mul_2Mulwhile/lstm_cell_1/Sigmoid_2:y:0&while/lstm_cell_1/Relu_1:activations:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_1/mul_2�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_1/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1�
while/IdentityIdentitywhile/add_1:z:0)^while/lstm_cell_1/BiasAdd/ReadVariableOp(^while/lstm_cell_1/MatMul/ReadVariableOp*^while/lstm_cell_1/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity�
while/Identity_1Identitywhile_while_maximum_iterations)^while/lstm_cell_1/BiasAdd/ReadVariableOp(^while/lstm_cell_1/MatMul/ReadVariableOp*^while/lstm_cell_1/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1�
while/Identity_2Identitywhile/add:z:0)^while/lstm_cell_1/BiasAdd/ReadVariableOp(^while/lstm_cell_1/MatMul/ReadVariableOp*^while/lstm_cell_1/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2�
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0)^while/lstm_cell_1/BiasAdd/ReadVariableOp(^while/lstm_cell_1/MatMul/ReadVariableOp*^while/lstm_cell_1/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3�
while/Identity_4Identitywhile/lstm_cell_1/mul_2:z:0)^while/lstm_cell_1/BiasAdd/ReadVariableOp(^while/lstm_cell_1/MatMul/ReadVariableOp*^while/lstm_cell_1/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:���������@2
while/Identity_4�
while/Identity_5Identitywhile/lstm_cell_1/add_1:z:0)^while/lstm_cell_1/BiasAdd/ReadVariableOp(^while/lstm_cell_1/MatMul/ReadVariableOp*^while/lstm_cell_1/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:���������@2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_1_biasadd_readvariableop_resource3while_lstm_cell_1_biasadd_readvariableop_resource_0"j
2while_lstm_cell_1_matmul_1_readvariableop_resource4while_lstm_cell_1_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_1_matmul_readvariableop_resource2while_lstm_cell_1_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :���������@:���������@: : :::2T
(while/lstm_cell_1/BiasAdd/ReadVariableOp(while/lstm_cell_1/BiasAdd/ReadVariableOp2R
'while/lstm_cell_1/MatMul/ReadVariableOp'while/lstm_cell_1/MatMul/ReadVariableOp2V
)while/lstm_cell_1/MatMul_1/ReadVariableOp)while/lstm_cell_1/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������@:-)
'
_output_shapes
:���������@:

_output_shapes
: :

_output_shapes
: 
�
�
-__inference_lstm_cell_2_layer_call_fn_1396078

inputs
states_0
states_1
unknown
	unknown_0
	unknown_1
identity

identity_1

identity_2��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:����������:����������:����������*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_lstm_cell_2_layer_call_and_return_conditional_losses_13927272
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity�

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity_1�

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*Z
_input_shapesI
G:���������@:����������:����������:::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs:RN
(
_output_shapes
:����������
"
_user_specified_name
states/0:RN
(
_output_shapes
:����������
"
_user_specified_name
states/1
�
�
while_cond_1393020
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1393020___redundant_placeholder05
1while_while_cond_1393020___redundant_placeholder15
1while_while_cond_1393020___redundant_placeholder25
1while_while_cond_1393020___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*U
_input_shapesD
B: : : : :����������:����������: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:����������:.*
(
_output_shapes
:����������:

_output_shapes
: :

_output_shapes
:
�
�
H__inference_lstm_cell_3_layer_call_and_return_conditional_losses_1396111

inputs
states_0
states_1"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1

identity_2��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	@�*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:����������2
add�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:���������@:���������@:���������@:���������@*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:���������@2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:���������@2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:���������@2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:���������@2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:���������@2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:���������@2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:���������@2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:���������@2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:���������@2
mul_2�
IdentityIdentity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:���������@2

Identity�

Identity_1Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:���������@2

Identity_1�

Identity_2Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:���������@2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*Y
_input_shapesH
F:����������:���������@:���������@:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:QM
'
_output_shapes
:���������@
"
_user_specified_name
states/0:QM
'
_output_shapes
:���������@
"
_user_specified_name
states/1
�
�
H__inference_lstm_cell_3_layer_call_and_return_conditional_losses_1393304

inputs

states
states_1"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1

identity_2��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	@�*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:����������2
add�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:���������@:���������@:���������@:���������@*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:���������@2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:���������@2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:���������@2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:���������@2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:���������@2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:���������@2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:���������@2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:���������@2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:���������@2
mul_2�
IdentityIdentity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:���������@2

Identity�

Identity_1Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:���������@2

Identity_1�

Identity_2Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:���������@2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*Y
_input_shapesH
F:����������:���������@:���������@:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������@
 
_user_specified_namestates:OK
'
_output_shapes
:���������@
 
_user_specified_namestates
�Y
�
D__inference_lstm_cc_layer_call_and_return_conditional_losses_1394619
inputs_0,
(lstm_cell_matmul_readvariableop_resource.
*lstm_cell_matmul_1_readvariableop_resource-
)lstm_cell_biasadd_readvariableop_resource
identity�� lstm_cell/BiasAdd/ReadVariableOp�lstm_cell/MatMul/ReadVariableOp�!lstm_cell/MatMul_1/ReadVariableOp�whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros/packed/1�
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������@2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/packed/1�
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:���������@2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm�
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :������������������\2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1�
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2
TensorArrayV2/element_shape�
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2�
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����\   27
5TensorArrayUnstack/TensorListFromTensor/element_shape�
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������\*
shrink_axis_mask2
strided_slice_2�
lstm_cell/MatMul/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	\�*
dtype02!
lstm_cell/MatMul/ReadVariableOp�
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell/MatMul�
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource*
_output_shapes
:	@�*
dtype02#
!lstm_cell/MatMul_1/ReadVariableOp�
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell/MatMul_1�
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
lstm_cell/add�
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp)lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02"
 lstm_cell/BiasAdd/ReadVariableOp�
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell/BiasAddd
lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Constx
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/split/split_dim�
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������@:���������@:���������@:���������@*
	num_split2
lstm_cell/split}
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*'
_output_shapes
:���������@2
lstm_cell/Sigmoid�
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*'
_output_shapes
:���������@2
lstm_cell/Sigmoid_1�
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������@2
lstm_cell/mult
lstm_cell/ReluRelulstm_cell/split:output:2*
T0*'
_output_shapes
:���������@2
lstm_cell/Relu�
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:���������@2
lstm_cell/mul_1�
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:���������@2
lstm_cell/add_1�
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*'
_output_shapes
:���������@2
lstm_cell/Sigmoid_2s
lstm_cell/Relu_1Relulstm_cell/add_1:z:0*
T0*'
_output_shapes
:���������@2
lstm_cell/Relu_1�
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:���������@2
lstm_cell/mul_2�
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   2
TensorArrayV2_1/element_shape�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter�
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0(lstm_cell_matmul_readvariableop_resource*lstm_cell_matmul_1_readvariableop_resource)lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������@:���������@: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_1394534*
condR
while_cond_1394533*K
output_shapes:
8: : : : :���������@:���������@: : : : : *
parallel_iterations 2
while�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   22
0TensorArrayV2Stack/TensorListStack/element_shape�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������@*
element_dtype02$
"TensorArrayV2Stack/TensorListStack�
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������@*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������@2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime�
IdentityIdentitystrided_slice_3:output:0!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:������������������\:::2D
 lstm_cell/BiasAdd/ReadVariableOp lstm_cell/BiasAdd/ReadVariableOp2B
lstm_cell/MatMul/ReadVariableOplstm_cell/MatMul/ReadVariableOp2F
!lstm_cell/MatMul_1/ReadVariableOp!lstm_cell/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :������������������\
"
_user_specified_name
inputs/0
�
�
(__inference_lstm_1_layer_call_fn_1395439
inputs_0
unknown
	unknown_0
	unknown_1
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:�������������������*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_lstm_1_layer_call_and_return_conditional_losses_13930902
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*5
_output_shapes#
!:�������������������2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:������������������@:::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :������������������@
"
_user_specified_name
inputs/0
�
�
while_cond_1395189
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1395189___redundant_placeholder05
1while_while_cond_1395189___redundant_placeholder15
1while_while_cond_1395189___redundant_placeholder25
1while_while_cond_1395189___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*U
_input_shapesD
B: : : : :����������:����������: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:����������:.*
(
_output_shapes
:����������:

_output_shapes
: :

_output_shapes
:
�
�
)__inference_lstm_cc_layer_call_fn_1394783
inputs_0
unknown
	unknown_0
	unknown_1
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_lstm_cc_layer_call_and_return_conditional_losses_13918702
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:������������������\:::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :������������������\
"
_user_specified_name
inputs/0
�B
�
while_body_1395190
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_06
2while_lstm_cell_2_matmul_readvariableop_resource_08
4while_lstm_cell_2_matmul_1_readvariableop_resource_07
3while_lstm_cell_2_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor4
0while_lstm_cell_2_matmul_readvariableop_resource6
2while_lstm_cell_2_matmul_1_readvariableop_resource5
1while_lstm_cell_2_biasadd_readvariableop_resource��(while/lstm_cell_2/BiasAdd/ReadVariableOp�'while/lstm_cell_2/MatMul/ReadVariableOp�)while/lstm_cell_2/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape�
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������@*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem�
'while/lstm_cell_2/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_2_matmul_readvariableop_resource_0*
_output_shapes
:	@�*
dtype02)
'while/lstm_cell_2/MatMul/ReadVariableOp�
while/lstm_cell_2/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_2/MatMul�
)while/lstm_cell_2/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_2_matmul_1_readvariableop_resource_0* 
_output_shapes
:
��*
dtype02+
)while/lstm_cell_2/MatMul_1/ReadVariableOp�
while/lstm_cell_2/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_2/MatMul_1�
while/lstm_cell_2/addAddV2"while/lstm_cell_2/MatMul:product:0$while/lstm_cell_2/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_2/add�
(while/lstm_cell_2/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_2_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02*
(while/lstm_cell_2/BiasAdd/ReadVariableOp�
while/lstm_cell_2/BiasAddBiasAddwhile/lstm_cell_2/add:z:00while/lstm_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_2/BiasAddt
while/lstm_cell_2/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_2/Const�
!while/lstm_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_2/split/split_dim�
while/lstm_cell_2/splitSplit*while/lstm_cell_2/split/split_dim:output:0"while/lstm_cell_2/BiasAdd:output:0*
T0*d
_output_shapesR
P:����������:����������:����������:����������*
	num_split2
while/lstm_cell_2/split�
while/lstm_cell_2/SigmoidSigmoid while/lstm_cell_2/split:output:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_2/Sigmoid�
while/lstm_cell_2/Sigmoid_1Sigmoid while/lstm_cell_2/split:output:1*
T0*(
_output_shapes
:����������2
while/lstm_cell_2/Sigmoid_1�
while/lstm_cell_2/mulMulwhile/lstm_cell_2/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:����������2
while/lstm_cell_2/mul�
while/lstm_cell_2/ReluRelu while/lstm_cell_2/split:output:2*
T0*(
_output_shapes
:����������2
while/lstm_cell_2/Relu�
while/lstm_cell_2/mul_1Mulwhile/lstm_cell_2/Sigmoid:y:0$while/lstm_cell_2/Relu:activations:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_2/mul_1�
while/lstm_cell_2/add_1AddV2while/lstm_cell_2/mul:z:0while/lstm_cell_2/mul_1:z:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_2/add_1�
while/lstm_cell_2/Sigmoid_2Sigmoid while/lstm_cell_2/split:output:3*
T0*(
_output_shapes
:����������2
while/lstm_cell_2/Sigmoid_2�
while/lstm_cell_2/Relu_1Reluwhile/lstm_cell_2/add_1:z:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_2/Relu_1�
while/lstm_cell_2/mul_2Mulwhile/lstm_cell_2/Sigmoid_2:y:0&while/lstm_cell_2/Relu_1:activations:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_2/mul_2�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_2/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1�
while/IdentityIdentitywhile/add_1:z:0)^while/lstm_cell_2/BiasAdd/ReadVariableOp(^while/lstm_cell_2/MatMul/ReadVariableOp*^while/lstm_cell_2/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity�
while/Identity_1Identitywhile_while_maximum_iterations)^while/lstm_cell_2/BiasAdd/ReadVariableOp(^while/lstm_cell_2/MatMul/ReadVariableOp*^while/lstm_cell_2/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1�
while/Identity_2Identitywhile/add:z:0)^while/lstm_cell_2/BiasAdd/ReadVariableOp(^while/lstm_cell_2/MatMul/ReadVariableOp*^while/lstm_cell_2/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2�
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0)^while/lstm_cell_2/BiasAdd/ReadVariableOp(^while/lstm_cell_2/MatMul/ReadVariableOp*^while/lstm_cell_2/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3�
while/Identity_4Identitywhile/lstm_cell_2/mul_2:z:0)^while/lstm_cell_2/BiasAdd/ReadVariableOp(^while/lstm_cell_2/MatMul/ReadVariableOp*^while/lstm_cell_2/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:����������2
while/Identity_4�
while/Identity_5Identitywhile/lstm_cell_2/add_1:z:0)^while/lstm_cell_2/BiasAdd/ReadVariableOp(^while/lstm_cell_2/MatMul/ReadVariableOp*^while/lstm_cell_2/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:����������2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_2_biasadd_readvariableop_resource3while_lstm_cell_2_biasadd_readvariableop_resource_0"j
2while_lstm_cell_2_matmul_1_readvariableop_resource4while_lstm_cell_2_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_2_matmul_readvariableop_resource2while_lstm_cell_2_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :����������:����������: : :::2T
(while/lstm_cell_2/BiasAdd/ReadVariableOp(while/lstm_cell_2/BiasAdd/ReadVariableOp2R
'while/lstm_cell_2/MatMul/ReadVariableOp'while/lstm_cell_2/MatMul/ReadVariableOp2V
)while/lstm_cell_2/MatMul_1/ReadVariableOp)while/lstm_cell_2/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:����������:.*
(
_output_shapes
:����������:

_output_shapes
: :

_output_shapes
: 
�D
�
C__inference_lstm_2_layer_call_and_return_conditional_losses_1393832

inputs
lstm_cell_3_1393750
lstm_cell_3_1393752
lstm_cell_3_1393754
identity��#lstm_cell_3/StatefulPartitionedCall�whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros/packed/1�
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������@2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/packed/1�
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:���������@2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm�
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:�������������������2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1�
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2
TensorArrayV2/element_shape�
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2�
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�����   27
5TensorArrayUnstack/TensorListFromTensor/element_shape�
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*
shrink_axis_mask2
strided_slice_2�
#lstm_cell_3/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_3_1393750lstm_cell_3_1393752lstm_cell_3_1393754*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������@:���������@:���������@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_lstm_cell_3_layer_call_and_return_conditional_losses_13933372%
#lstm_cell_3/StatefulPartitionedCall�
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   2
TensorArrayV2_1/element_shape�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter�
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_3_1393750lstm_cell_3_1393752lstm_cell_3_1393754*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������@:���������@: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_1393763*
condR
while_cond_1393762*K
output_shapes:
8: : : : :���������@:���������@: : : : : *
parallel_iterations 2
while�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   22
0TensorArrayV2Stack/TensorListStack/element_shape�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������@*
element_dtype02$
"TensorArrayV2Stack/TensorListStack�
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������@*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������@2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime�
IdentityIdentitystrided_slice_3:output:0$^lstm_cell_3/StatefulPartitionedCall^while*
T0*'
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:�������������������:::2J
#lstm_cell_3/StatefulPartitionedCall#lstm_cell_3/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:�������������������
 
_user_specified_nameinputs
�
�
while_cond_1394861
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1394861___redundant_placeholder05
1while_while_cond_1394861___redundant_placeholder15
1while_while_cond_1394861___redundant_placeholder25
1while_while_cond_1394861___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :���������@:���������@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������@:-)
'
_output_shapes
:���������@:

_output_shapes
: :

_output_shapes
:
�D
�
D__inference_lstm_cc_layer_call_and_return_conditional_losses_1392002

inputs
lstm_cell_1391920
lstm_cell_1391922
lstm_cell_1391924
identity��!lstm_cell/StatefulPartitionedCall�whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros/packed/1�
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������@2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/packed/1�
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:���������@2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm�
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :������������������\2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1�
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2
TensorArrayV2/element_shape�
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2�
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����\   27
5TensorArrayUnstack/TensorListFromTensor/element_shape�
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������\*
shrink_axis_mask2
strided_slice_2�
!lstm_cell/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_1391920lstm_cell_1391922lstm_cell_1391924*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������@:���������@:���������@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_lstm_cell_layer_call_and_return_conditional_losses_13915072#
!lstm_cell/StatefulPartitionedCall�
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   2
TensorArrayV2_1/element_shape�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter�
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_1391920lstm_cell_1391922lstm_cell_1391924*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������@:���������@: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_1391933*
condR
while_cond_1391932*K
output_shapes:
8: : : : :���������@:���������@: : : : : *
parallel_iterations 2
while�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   22
0TensorArrayV2Stack/TensorListStack/element_shape�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������@*
element_dtype02$
"TensorArrayV2Stack/TensorListStack�
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������@*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������@2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime�
IdentityIdentitystrided_slice_3:output:0"^lstm_cell/StatefulPartitionedCall^while*
T0*'
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:������������������\:::2F
!lstm_cell/StatefulPartitionedCall!lstm_cell/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :������������������\
 
_user_specified_nameinputs
�
�
while_cond_1392542
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1392542___redundant_placeholder05
1while_while_cond_1392542___redundant_placeholder15
1while_while_cond_1392542___redundant_placeholder25
1while_while_cond_1392542___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :���������@:���������@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������@:-)
'
_output_shapes
:���������@:

_output_shapes
: :

_output_shapes
:
�Z
�
A__inference_lstm_layer_call_and_return_conditional_losses_1395100
inputs_0.
*lstm_cell_1_matmul_readvariableop_resource0
,lstm_cell_1_matmul_1_readvariableop_resource/
+lstm_cell_1_biasadd_readvariableop_resource
identity��"lstm_cell_1/BiasAdd/ReadVariableOp�!lstm_cell_1/MatMul/ReadVariableOp�#lstm_cell_1/MatMul_1/ReadVariableOp�whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros/packed/1�
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������@2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/packed/1�
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:���������@2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm�
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :������������������\2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1�
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2
TensorArrayV2/element_shape�
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2�
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����\   27
5TensorArrayUnstack/TensorListFromTensor/element_shape�
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������\*
shrink_axis_mask2
strided_slice_2�
!lstm_cell_1/MatMul/ReadVariableOpReadVariableOp*lstm_cell_1_matmul_readvariableop_resource*
_output_shapes
:	\�*
dtype02#
!lstm_cell_1/MatMul/ReadVariableOp�
lstm_cell_1/MatMulMatMulstrided_slice_2:output:0)lstm_cell_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_1/MatMul�
#lstm_cell_1/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_1_matmul_1_readvariableop_resource*
_output_shapes
:	@�*
dtype02%
#lstm_cell_1/MatMul_1/ReadVariableOp�
lstm_cell_1/MatMul_1MatMulzeros:output:0+lstm_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_1/MatMul_1�
lstm_cell_1/addAddV2lstm_cell_1/MatMul:product:0lstm_cell_1/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
lstm_cell_1/add�
"lstm_cell_1/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02$
"lstm_cell_1/BiasAdd/ReadVariableOp�
lstm_cell_1/BiasAddBiasAddlstm_cell_1/add:z:0*lstm_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_1/BiasAddh
lstm_cell_1/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_1/Const|
lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_1/split/split_dim�
lstm_cell_1/splitSplit$lstm_cell_1/split/split_dim:output:0lstm_cell_1/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������@:���������@:���������@:���������@*
	num_split2
lstm_cell_1/split�
lstm_cell_1/SigmoidSigmoidlstm_cell_1/split:output:0*
T0*'
_output_shapes
:���������@2
lstm_cell_1/Sigmoid�
lstm_cell_1/Sigmoid_1Sigmoidlstm_cell_1/split:output:1*
T0*'
_output_shapes
:���������@2
lstm_cell_1/Sigmoid_1�
lstm_cell_1/mulMullstm_cell_1/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������@2
lstm_cell_1/mulz
lstm_cell_1/ReluRelulstm_cell_1/split:output:2*
T0*'
_output_shapes
:���������@2
lstm_cell_1/Relu�
lstm_cell_1/mul_1Mullstm_cell_1/Sigmoid:y:0lstm_cell_1/Relu:activations:0*
T0*'
_output_shapes
:���������@2
lstm_cell_1/mul_1�
lstm_cell_1/add_1AddV2lstm_cell_1/mul:z:0lstm_cell_1/mul_1:z:0*
T0*'
_output_shapes
:���������@2
lstm_cell_1/add_1�
lstm_cell_1/Sigmoid_2Sigmoidlstm_cell_1/split:output:3*
T0*'
_output_shapes
:���������@2
lstm_cell_1/Sigmoid_2y
lstm_cell_1/Relu_1Relulstm_cell_1/add_1:z:0*
T0*'
_output_shapes
:���������@2
lstm_cell_1/Relu_1�
lstm_cell_1/mul_2Mullstm_cell_1/Sigmoid_2:y:0 lstm_cell_1/Relu_1:activations:0*
T0*'
_output_shapes
:���������@2
lstm_cell_1/mul_2�
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   2
TensorArrayV2_1/element_shape�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter�
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_1_matmul_readvariableop_resource,lstm_cell_1_matmul_1_readvariableop_resource+lstm_cell_1_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������@:���������@: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_1395015*
condR
while_cond_1395014*K
output_shapes:
8: : : : :���������@:���������@: : : : : *
parallel_iterations 2
while�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   22
0TensorArrayV2Stack/TensorListStack/element_shape�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������@*
element_dtype02$
"TensorArrayV2Stack/TensorListStack�
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������@*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������@2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime�
IdentityIdentitytranspose_1:y:0#^lstm_cell_1/BiasAdd/ReadVariableOp"^lstm_cell_1/MatMul/ReadVariableOp$^lstm_cell_1/MatMul_1/ReadVariableOp^while*
T0*4
_output_shapes"
 :������������������@2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:������������������\:::2H
"lstm_cell_1/BiasAdd/ReadVariableOp"lstm_cell_1/BiasAdd/ReadVariableOp2F
!lstm_cell_1/MatMul/ReadVariableOp!lstm_cell_1/MatMul/ReadVariableOp2J
#lstm_cell_1/MatMul_1/ReadVariableOp#lstm_cell_1/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :������������������\
"
_user_specified_name
inputs/0
�
�
H__inference_lstm_cell_1_layer_call_and_return_conditional_losses_1395911

inputs
states_0
states_1"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1

identity_2��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	\�*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	@�*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:����������2
add�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:���������@:���������@:���������@:���������@*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:���������@2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:���������@2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:���������@2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:���������@2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:���������@2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:���������@2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:���������@2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:���������@2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:���������@2
mul_2�
IdentityIdentity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:���������@2

Identity�

Identity_1Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:���������@2

Identity_1�

Identity_2Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:���������@2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:���������\:���������@:���������@:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:���������\
 
_user_specified_nameinputs:QM
'
_output_shapes
:���������@
"
_user_specified_name
states/0:QM
'
_output_shapes
:���������@
"
_user_specified_name
states/1
�D
�
A__inference_lstm_layer_call_and_return_conditional_losses_1392480

inputs
lstm_cell_1_1392398
lstm_cell_1_1392400
lstm_cell_1_1392402
identity��#lstm_cell_1/StatefulPartitionedCall�whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros/packed/1�
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������@2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/packed/1�
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:���������@2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm�
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :������������������\2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1�
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2
TensorArrayV2/element_shape�
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2�
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����\   27
5TensorArrayUnstack/TensorListFromTensor/element_shape�
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������\*
shrink_axis_mask2
strided_slice_2�
#lstm_cell_1/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_1_1392398lstm_cell_1_1392400lstm_cell_1_1392402*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������@:���������@:���������@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_lstm_cell_1_layer_call_and_return_conditional_losses_13920842%
#lstm_cell_1/StatefulPartitionedCall�
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   2
TensorArrayV2_1/element_shape�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter�
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_1_1392398lstm_cell_1_1392400lstm_cell_1_1392402*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������@:���������@: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_1392411*
condR
while_cond_1392410*K
output_shapes:
8: : : : :���������@:���������@: : : : : *
parallel_iterations 2
while�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   22
0TensorArrayV2Stack/TensorListStack/element_shape�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������@*
element_dtype02$
"TensorArrayV2Stack/TensorListStack�
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������@*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������@2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime�
IdentityIdentitytranspose_1:y:0$^lstm_cell_1/StatefulPartitionedCall^while*
T0*4
_output_shapes"
 :������������������@2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:������������������\:::2J
#lstm_cell_1/StatefulPartitionedCall#lstm_cell_1/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :������������������\
 
_user_specified_nameinputs
�
�
H__inference_lstm_cell_2_layer_call_and_return_conditional_losses_1396011

inputs
states_0
states_1"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1

identity_2��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@�*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
��*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:����������2
add�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:����������:����������:����������:����������*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:����������2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:����������2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:����������2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:����������2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:����������2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:����������2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:����������2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:����������2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:����������2
mul_2�
IdentityIdentity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity�

Identity_1Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity_1�

Identity_2Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*Z
_input_shapesI
G:���������@:����������:����������:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs:RN
(
_output_shapes
:����������
"
_user_specified_name
states/0:RN
(
_output_shapes
:����������
"
_user_specified_name
states/1
�
�
-__inference_lstm_cell_3_layer_call_fn_1396161

inputs
states_0
states_1
unknown
	unknown_0
	unknown_1
identity

identity_1

identity_2��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������@:���������@:���������@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_lstm_cell_3_layer_call_and_return_conditional_losses_13933042
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������@2

Identity�

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:���������@2

Identity_1�

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:���������@2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*Y
_input_shapesH
F:����������:���������@:���������@:::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:QM
'
_output_shapes
:���������@
"
_user_specified_name
states/0:QM
'
_output_shapes
:���������@
"
_user_specified_name
states/1
�D
�
D__inference_lstm_cc_layer_call_and_return_conditional_losses_1391870

inputs
lstm_cell_1391788
lstm_cell_1391790
lstm_cell_1391792
identity��!lstm_cell/StatefulPartitionedCall�whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros/packed/1�
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������@2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/packed/1�
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:���������@2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm�
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :������������������\2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1�
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2
TensorArrayV2/element_shape�
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2�
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����\   27
5TensorArrayUnstack/TensorListFromTensor/element_shape�
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������\*
shrink_axis_mask2
strided_slice_2�
!lstm_cell/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_1391788lstm_cell_1391790lstm_cell_1391792*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������@:���������@:���������@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_lstm_cell_layer_call_and_return_conditional_losses_13914742#
!lstm_cell/StatefulPartitionedCall�
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   2
TensorArrayV2_1/element_shape�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter�
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_1391788lstm_cell_1391790lstm_cell_1391792*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������@:���������@: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_1391801*
condR
while_cond_1391800*K
output_shapes:
8: : : : :���������@:���������@: : : : : *
parallel_iterations 2
while�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   22
0TensorArrayV2Stack/TensorListStack/element_shape�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������@*
element_dtype02$
"TensorArrayV2Stack/TensorListStack�
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������@*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������@2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime�
IdentityIdentitystrided_slice_3:output:0"^lstm_cell/StatefulPartitionedCall^while*
T0*'
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:������������������\:::2F
!lstm_cell/StatefulPartitionedCall!lstm_cell/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :������������������\
 
_user_specified_nameinputs
�%
�
while_body_1392543
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_lstm_cell_1_1392567_0
while_lstm_cell_1_1392569_0
while_lstm_cell_1_1392571_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_lstm_cell_1_1392567
while_lstm_cell_1_1392569
while_lstm_cell_1_1392571��)while/lstm_cell_1/StatefulPartitionedCall�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����\   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape�
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������\*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem�
)while/lstm_cell_1/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_1_1392567_0while_lstm_cell_1_1392569_0while_lstm_cell_1_1392571_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������@:���������@:���������@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_lstm_cell_1_layer_call_and_return_conditional_losses_13921172+
)while/lstm_cell_1/StatefulPartitionedCall�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/lstm_cell_1/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1�
while/IdentityIdentitywhile/add_1:z:0*^while/lstm_cell_1/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity�
while/Identity_1Identitywhile_while_maximum_iterations*^while/lstm_cell_1/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1�
while/Identity_2Identitywhile/add:z:0*^while/lstm_cell_1/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2�
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*^while/lstm_cell_1/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3�
while/Identity_4Identity2while/lstm_cell_1/StatefulPartitionedCall:output:1*^while/lstm_cell_1/StatefulPartitionedCall*
T0*'
_output_shapes
:���������@2
while/Identity_4�
while/Identity_5Identity2while/lstm_cell_1/StatefulPartitionedCall:output:2*^while/lstm_cell_1/StatefulPartitionedCall*
T0*'
_output_shapes
:���������@2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"8
while_lstm_cell_1_1392567while_lstm_cell_1_1392567_0"8
while_lstm_cell_1_1392569while_lstm_cell_1_1392569_0"8
while_lstm_cell_1_1392571while_lstm_cell_1_1392571_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :���������@:���������@: : :::2V
)while/lstm_cell_1/StatefulPartitionedCall)while/lstm_cell_1/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������@:-)
'
_output_shapes
:���������@:

_output_shapes
: :

_output_shapes
: 
�
�
while_cond_1394686
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1394686___redundant_placeholder05
1while_while_cond_1394686___redundant_placeholder15
1while_while_cond_1394686___redundant_placeholder25
1while_while_cond_1394686___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :���������@:���������@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������@:-)
'
_output_shapes
:���������@:

_output_shapes
: :

_output_shapes
:
�[
�
C__inference_lstm_1_layer_call_and_return_conditional_losses_1395428
inputs_0.
*lstm_cell_2_matmul_readvariableop_resource0
,lstm_cell_2_matmul_1_readvariableop_resource/
+lstm_cell_2_biasadd_readvariableop_resource
identity��"lstm_cell_2/BiasAdd/ReadVariableOp�!lstm_cell_2/MatMul/ReadVariableOp�#lstm_cell_2/MatMul_1/ReadVariableOp�whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :�2
zeros/packed/1�
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:����������2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :�2
zeros_1/packed/1�
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:����������2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm�
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :������������������@2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1�
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2
TensorArrayV2/element_shape�
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2�
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   27
5TensorArrayUnstack/TensorListFromTensor/element_shape�
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������@*
shrink_axis_mask2
strided_slice_2�
!lstm_cell_2/MatMul/ReadVariableOpReadVariableOp*lstm_cell_2_matmul_readvariableop_resource*
_output_shapes
:	@�*
dtype02#
!lstm_cell_2/MatMul/ReadVariableOp�
lstm_cell_2/MatMulMatMulstrided_slice_2:output:0)lstm_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_2/MatMul�
#lstm_cell_2/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_2_matmul_1_readvariableop_resource* 
_output_shapes
:
��*
dtype02%
#lstm_cell_2/MatMul_1/ReadVariableOp�
lstm_cell_2/MatMul_1MatMulzeros:output:0+lstm_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_2/MatMul_1�
lstm_cell_2/addAddV2lstm_cell_2/MatMul:product:0lstm_cell_2/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
lstm_cell_2/add�
"lstm_cell_2/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02$
"lstm_cell_2/BiasAdd/ReadVariableOp�
lstm_cell_2/BiasAddBiasAddlstm_cell_2/add:z:0*lstm_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_2/BiasAddh
lstm_cell_2/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_2/Const|
lstm_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_2/split/split_dim�
lstm_cell_2/splitSplit$lstm_cell_2/split/split_dim:output:0lstm_cell_2/BiasAdd:output:0*
T0*d
_output_shapesR
P:����������:����������:����������:����������*
	num_split2
lstm_cell_2/split�
lstm_cell_2/SigmoidSigmoidlstm_cell_2/split:output:0*
T0*(
_output_shapes
:����������2
lstm_cell_2/Sigmoid�
lstm_cell_2/Sigmoid_1Sigmoidlstm_cell_2/split:output:1*
T0*(
_output_shapes
:����������2
lstm_cell_2/Sigmoid_1�
lstm_cell_2/mulMullstm_cell_2/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:����������2
lstm_cell_2/mul{
lstm_cell_2/ReluRelulstm_cell_2/split:output:2*
T0*(
_output_shapes
:����������2
lstm_cell_2/Relu�
lstm_cell_2/mul_1Mullstm_cell_2/Sigmoid:y:0lstm_cell_2/Relu:activations:0*
T0*(
_output_shapes
:����������2
lstm_cell_2/mul_1�
lstm_cell_2/add_1AddV2lstm_cell_2/mul:z:0lstm_cell_2/mul_1:z:0*
T0*(
_output_shapes
:����������2
lstm_cell_2/add_1�
lstm_cell_2/Sigmoid_2Sigmoidlstm_cell_2/split:output:3*
T0*(
_output_shapes
:����������2
lstm_cell_2/Sigmoid_2z
lstm_cell_2/Relu_1Relulstm_cell_2/add_1:z:0*
T0*(
_output_shapes
:����������2
lstm_cell_2/Relu_1�
lstm_cell_2/mul_2Mullstm_cell_2/Sigmoid_2:y:0 lstm_cell_2/Relu_1:activations:0*
T0*(
_output_shapes
:����������2
lstm_cell_2/mul_2�
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�����   2
TensorArrayV2_1/element_shape�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter�
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_2_matmul_readvariableop_resource,lstm_cell_2_matmul_1_readvariableop_resource+lstm_cell_2_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :����������:����������: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_1395343*
condR
while_cond_1395342*M
output_shapes<
:: : : : :����������:����������: : : : : *
parallel_iterations 2
while�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�����   22
0TensorArrayV2Stack/TensorListStack/element_shape�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:�������������������*
element_dtype02$
"TensorArrayV2Stack/TensorListStack�
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:�������������������2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime�
IdentityIdentitytranspose_1:y:0#^lstm_cell_2/BiasAdd/ReadVariableOp"^lstm_cell_2/MatMul/ReadVariableOp$^lstm_cell_2/MatMul_1/ReadVariableOp^while*
T0*5
_output_shapes#
!:�������������������2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:������������������@:::2H
"lstm_cell_2/BiasAdd/ReadVariableOp"lstm_cell_2/BiasAdd/ReadVariableOp2F
!lstm_cell_2/MatMul/ReadVariableOp!lstm_cell_2/MatMul/ReadVariableOp2J
#lstm_cell_2/MatMul_1/ReadVariableOp#lstm_cell_2/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :������������������@
"
_user_specified_name
inputs/0
ٰ
�
__inference_call_1394466
input_ac
input_cc3
/lstm_lstm_cell_1_matmul_readvariableop_resource5
1lstm_lstm_cell_1_matmul_1_readvariableop_resource4
0lstm_lstm_cell_1_biasadd_readvariableop_resource5
1lstm_1_lstm_cell_2_matmul_readvariableop_resource7
3lstm_1_lstm_cell_2_matmul_1_readvariableop_resource6
2lstm_1_lstm_cell_2_biasadd_readvariableop_resource5
1lstm_2_lstm_cell_3_matmul_readvariableop_resource7
3lstm_2_lstm_cell_3_matmul_1_readvariableop_resource6
2lstm_2_lstm_cell_3_biasadd_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource4
0lstm_cc_lstm_cell_matmul_readvariableop_resource6
2lstm_cc_lstm_cell_matmul_1_readvariableop_resource5
1lstm_cc_lstm_cell_biasadd_readvariableop_resource+
'dense_cc_matmul_readvariableop_resource,
(dense_cc_biasadd_readvariableop_resource
identity

identity_1��dense/BiasAdd/ReadVariableOp�dense/MatMul/ReadVariableOp�dense_1/BiasAdd/ReadVariableOp�dense_1/MatMul/ReadVariableOp�dense_cc/BiasAdd/ReadVariableOp�dense_cc/MatMul/ReadVariableOp�'lstm/lstm_cell_1/BiasAdd/ReadVariableOp�&lstm/lstm_cell_1/MatMul/ReadVariableOp�(lstm/lstm_cell_1/MatMul_1/ReadVariableOp�
lstm/while�)lstm_1/lstm_cell_2/BiasAdd/ReadVariableOp�(lstm_1/lstm_cell_2/MatMul/ReadVariableOp�*lstm_1/lstm_cell_2/MatMul_1/ReadVariableOp�lstm_1/while�)lstm_2/lstm_cell_3/BiasAdd/ReadVariableOp�(lstm_2/lstm_cell_3/MatMul/ReadVariableOp�*lstm_2/lstm_cell_3/MatMul_1/ReadVariableOp�lstm_2/while�(lstm_cc/lstm_cell/BiasAdd/ReadVariableOp�'lstm_cc/lstm_cell/MatMul/ReadVariableOp�)lstm_cc/lstm_cell/MatMul_1/ReadVariableOp�lstm_cc/whileZ
CastCastinput_ac*

DstT0*

SrcT0*"
_output_shapes
:2\2
Castm

lstm/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"2      \   2

lstm/Shape~
lstm/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm/strided_slice/stack�
lstm/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice/stack_1�
lstm/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice/stack_2�
lstm/strided_sliceStridedSlicelstm/Shape:output:0!lstm/strided_slice/stack:output:0#lstm/strided_slice/stack_1:output:0#lstm/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm/strided_slicef
lstm/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
lstm/zeros/mul/y�
lstm/zeros/mulMullstm/strided_slice:output:0lstm/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm/zeros/muli
lstm/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
lstm/zeros/Less/y{
lstm/zeros/LessLesslstm/zeros/mul:z:0lstm/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm/zeros/Lessl
lstm/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
lstm/zeros/packed/1�
lstm/zeros/packedPacklstm/strided_slice:output:0lstm/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm/zeros/packedi
lstm/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm/zeros/Const�

lstm/zerosFilllstm/zeros/packed:output:0lstm/zeros/Const:output:0*
T0*
_output_shapes

:2@2

lstm/zerosj
lstm/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
lstm/zeros_1/mul/y�
lstm/zeros_1/mulMullstm/strided_slice:output:0lstm/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm/zeros_1/mulm
lstm/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
lstm/zeros_1/Less/y�
lstm/zeros_1/LessLesslstm/zeros_1/mul:z:0lstm/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm/zeros_1/Lessp
lstm/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
lstm/zeros_1/packed/1�
lstm/zeros_1/packedPacklstm/strided_slice:output:0lstm/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm/zeros_1/packedm
lstm/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm/zeros_1/Const�
lstm/zeros_1Filllstm/zeros_1/packed:output:0lstm/zeros_1/Const:output:0*
T0*
_output_shapes

:2@2
lstm/zeros_1
lstm/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm/transpose/perm�
lstm/transpose	TransposeCast:y:0lstm/transpose/perm:output:0*
T0*"
_output_shapes
:2\2
lstm/transposeq
lstm/Shape_1Const*
_output_shapes
:*
dtype0*!
valueB"   2   \   2
lstm/Shape_1�
lstm/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm/strided_slice_1/stack�
lstm/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice_1/stack_1�
lstm/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice_1/stack_2�
lstm/strided_slice_1StridedSlicelstm/Shape_1:output:0#lstm/strided_slice_1/stack:output:0%lstm/strided_slice_1/stack_1:output:0%lstm/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm/strided_slice_1�
 lstm/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2"
 lstm/TensorArrayV2/element_shape�
lstm/TensorArrayV2TensorListReserve)lstm/TensorArrayV2/element_shape:output:0lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm/TensorArrayV2�
:lstm/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"2   \   2<
:lstm/TensorArrayUnstack/TensorListFromTensor/element_shape�
,lstm/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm/transpose:y:0Clstm/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02.
,lstm/TensorArrayUnstack/TensorListFromTensor�
lstm/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm/strided_slice_2/stack�
lstm/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice_2/stack_1�
lstm/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice_2/stack_2�
lstm/strided_slice_2StridedSlicelstm/transpose:y:0#lstm/strided_slice_2/stack:output:0%lstm/strided_slice_2/stack_1:output:0%lstm/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:2\*
shrink_axis_mask2
lstm/strided_slice_2�
&lstm/lstm_cell_1/MatMul/ReadVariableOpReadVariableOp/lstm_lstm_cell_1_matmul_readvariableop_resource*
_output_shapes
:	\�*
dtype02(
&lstm/lstm_cell_1/MatMul/ReadVariableOp�
lstm/lstm_cell_1/MatMulMatMullstm/strided_slice_2:output:0.lstm/lstm_cell_1/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2�2
lstm/lstm_cell_1/MatMul�
(lstm/lstm_cell_1/MatMul_1/ReadVariableOpReadVariableOp1lstm_lstm_cell_1_matmul_1_readvariableop_resource*
_output_shapes
:	@�*
dtype02*
(lstm/lstm_cell_1/MatMul_1/ReadVariableOp�
lstm/lstm_cell_1/MatMul_1MatMullstm/zeros:output:00lstm/lstm_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	2�2
lstm/lstm_cell_1/MatMul_1�
lstm/lstm_cell_1/addAddV2!lstm/lstm_cell_1/MatMul:product:0#lstm/lstm_cell_1/MatMul_1:product:0*
T0*
_output_shapes
:	2�2
lstm/lstm_cell_1/add�
'lstm/lstm_cell_1/BiasAdd/ReadVariableOpReadVariableOp0lstm_lstm_cell_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02)
'lstm/lstm_cell_1/BiasAdd/ReadVariableOp�
lstm/lstm_cell_1/BiasAddBiasAddlstm/lstm_cell_1/add:z:0/lstm/lstm_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	2�2
lstm/lstm_cell_1/BiasAddr
lstm/lstm_cell_1/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm/lstm_cell_1/Const�
 lstm/lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 lstm/lstm_cell_1/split/split_dim�
lstm/lstm_cell_1/splitSplit)lstm/lstm_cell_1/split/split_dim:output:0!lstm/lstm_cell_1/BiasAdd:output:0*
T0*<
_output_shapes*
(:2@:2@:2@:2@*
	num_split2
lstm/lstm_cell_1/split�
lstm/lstm_cell_1/SigmoidSigmoidlstm/lstm_cell_1/split:output:0*
T0*
_output_shapes

:2@2
lstm/lstm_cell_1/Sigmoid�
lstm/lstm_cell_1/Sigmoid_1Sigmoidlstm/lstm_cell_1/split:output:1*
T0*
_output_shapes

:2@2
lstm/lstm_cell_1/Sigmoid_1�
lstm/lstm_cell_1/mulMullstm/lstm_cell_1/Sigmoid_1:y:0lstm/zeros_1:output:0*
T0*
_output_shapes

:2@2
lstm/lstm_cell_1/mul�
lstm/lstm_cell_1/ReluRelulstm/lstm_cell_1/split:output:2*
T0*
_output_shapes

:2@2
lstm/lstm_cell_1/Relu�
lstm/lstm_cell_1/mul_1Mullstm/lstm_cell_1/Sigmoid:y:0#lstm/lstm_cell_1/Relu:activations:0*
T0*
_output_shapes

:2@2
lstm/lstm_cell_1/mul_1�
lstm/lstm_cell_1/add_1AddV2lstm/lstm_cell_1/mul:z:0lstm/lstm_cell_1/mul_1:z:0*
T0*
_output_shapes

:2@2
lstm/lstm_cell_1/add_1�
lstm/lstm_cell_1/Sigmoid_2Sigmoidlstm/lstm_cell_1/split:output:3*
T0*
_output_shapes

:2@2
lstm/lstm_cell_1/Sigmoid_2
lstm/lstm_cell_1/Relu_1Relulstm/lstm_cell_1/add_1:z:0*
T0*
_output_shapes

:2@2
lstm/lstm_cell_1/Relu_1�
lstm/lstm_cell_1/mul_2Mullstm/lstm_cell_1/Sigmoid_2:y:0%lstm/lstm_cell_1/Relu_1:activations:0*
T0*
_output_shapes

:2@2
lstm/lstm_cell_1/mul_2�
"lstm/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"2   @   2$
"lstm/TensorArrayV2_1/element_shape�
lstm/TensorArrayV2_1TensorListReserve+lstm/TensorArrayV2_1/element_shape:output:0lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm/TensorArrayV2_1X
	lstm/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
	lstm/time�
lstm/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2
lstm/while/maximum_iterationst
lstm/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm/while/loop_counter�

lstm/whileWhile lstm/while/loop_counter:output:0&lstm/while/maximum_iterations:output:0lstm/time:output:0lstm/TensorArrayV2_1:handle:0lstm/zeros:output:0lstm/zeros_1:output:0lstm/strided_slice_1:output:0<lstm/TensorArrayUnstack/TensorListFromTensor:output_handle:0/lstm_lstm_cell_1_matmul_readvariableop_resource1lstm_lstm_cell_1_matmul_1_readvariableop_resource0lstm_lstm_cell_1_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*:
_output_shapes(
&: : : : :2@:2@: : : : : *%
_read_only_resource_inputs
	
*#
bodyR
lstm_while_body_1393911*#
condR
lstm_while_cond_1393910*9
output_shapes(
&: : : : :2@:2@: : : : : *
parallel_iterations 2

lstm/while�
5lstm/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"2   @   27
5lstm/TensorArrayV2Stack/TensorListStack/element_shape�
'lstm/TensorArrayV2Stack/TensorListStackTensorListStacklstm/while:output:3>lstm/TensorArrayV2Stack/TensorListStack/element_shape:output:0*"
_output_shapes
:2@*
element_dtype02)
'lstm/TensorArrayV2Stack/TensorListStack�
lstm/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
lstm/strided_slice_3/stack�
lstm/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
lstm/strided_slice_3/stack_1�
lstm/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice_3/stack_2�
lstm/strided_slice_3StridedSlice0lstm/TensorArrayV2Stack/TensorListStack:tensor:0#lstm/strided_slice_3/stack:output:0%lstm/strided_slice_3/stack_1:output:0%lstm/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:2@*
shrink_axis_mask2
lstm/strided_slice_3�
lstm/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm/transpose_1/perm�
lstm/transpose_1	Transpose0lstm/TensorArrayV2Stack/TensorListStack:tensor:0lstm/transpose_1/perm:output:0*
T0*"
_output_shapes
:2@2
lstm/transpose_1p
lstm/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm/runtimeq
lstm_1/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"2      @   2
lstm_1/Shape�
lstm_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_1/strided_slice/stack�
lstm_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_1/strided_slice/stack_1�
lstm_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_1/strided_slice/stack_2�
lstm_1/strided_sliceStridedSlicelstm_1/Shape:output:0#lstm_1/strided_slice/stack:output:0%lstm_1/strided_slice/stack_1:output:0%lstm_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_1/strided_slicek
lstm_1/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :�2
lstm_1/zeros/mul/y�
lstm_1/zeros/mulMullstm_1/strided_slice:output:0lstm_1/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_1/zeros/mulm
lstm_1/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
lstm_1/zeros/Less/y�
lstm_1/zeros/LessLesslstm_1/zeros/mul:z:0lstm_1/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_1/zeros/Lessq
lstm_1/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :�2
lstm_1/zeros/packed/1�
lstm_1/zeros/packedPacklstm_1/strided_slice:output:0lstm_1/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_1/zeros/packedm
lstm_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_1/zeros/Const�
lstm_1/zerosFilllstm_1/zeros/packed:output:0lstm_1/zeros/Const:output:0*
T0*
_output_shapes
:	2�2
lstm_1/zeroso
lstm_1/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :�2
lstm_1/zeros_1/mul/y�
lstm_1/zeros_1/mulMullstm_1/strided_slice:output:0lstm_1/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_1/zeros_1/mulq
lstm_1/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
lstm_1/zeros_1/Less/y�
lstm_1/zeros_1/LessLesslstm_1/zeros_1/mul:z:0lstm_1/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_1/zeros_1/Lessu
lstm_1/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :�2
lstm_1/zeros_1/packed/1�
lstm_1/zeros_1/packedPacklstm_1/strided_slice:output:0 lstm_1/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_1/zeros_1/packedq
lstm_1/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_1/zeros_1/Const�
lstm_1/zeros_1Filllstm_1/zeros_1/packed:output:0lstm_1/zeros_1/Const:output:0*
T0*
_output_shapes
:	2�2
lstm_1/zeros_1�
lstm_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_1/transpose/perm�
lstm_1/transpose	Transposelstm/transpose_1:y:0lstm_1/transpose/perm:output:0*
T0*"
_output_shapes
:2@2
lstm_1/transposeu
lstm_1/Shape_1Const*
_output_shapes
:*
dtype0*!
valueB"   2   @   2
lstm_1/Shape_1�
lstm_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_1/strided_slice_1/stack�
lstm_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_1/strided_slice_1/stack_1�
lstm_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_1/strided_slice_1/stack_2�
lstm_1/strided_slice_1StridedSlicelstm_1/Shape_1:output:0%lstm_1/strided_slice_1/stack:output:0'lstm_1/strided_slice_1/stack_1:output:0'lstm_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_1/strided_slice_1�
"lstm_1/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2$
"lstm_1/TensorArrayV2/element_shape�
lstm_1/TensorArrayV2TensorListReserve+lstm_1/TensorArrayV2/element_shape:output:0lstm_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_1/TensorArrayV2�
<lstm_1/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"2   @   2>
<lstm_1/TensorArrayUnstack/TensorListFromTensor/element_shape�
.lstm_1/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_1/transpose:y:0Elstm_1/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type020
.lstm_1/TensorArrayUnstack/TensorListFromTensor�
lstm_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_1/strided_slice_2/stack�
lstm_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_1/strided_slice_2/stack_1�
lstm_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_1/strided_slice_2/stack_2�
lstm_1/strided_slice_2StridedSlicelstm_1/transpose:y:0%lstm_1/strided_slice_2/stack:output:0'lstm_1/strided_slice_2/stack_1:output:0'lstm_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:2@*
shrink_axis_mask2
lstm_1/strided_slice_2�
(lstm_1/lstm_cell_2/MatMul/ReadVariableOpReadVariableOp1lstm_1_lstm_cell_2_matmul_readvariableop_resource*
_output_shapes
:	@�*
dtype02*
(lstm_1/lstm_cell_2/MatMul/ReadVariableOp�
lstm_1/lstm_cell_2/MatMulMatMullstm_1/strided_slice_2:output:00lstm_1/lstm_cell_2/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2�2
lstm_1/lstm_cell_2/MatMul�
*lstm_1/lstm_cell_2/MatMul_1/ReadVariableOpReadVariableOp3lstm_1_lstm_cell_2_matmul_1_readvariableop_resource* 
_output_shapes
:
��*
dtype02,
*lstm_1/lstm_cell_2/MatMul_1/ReadVariableOp�
lstm_1/lstm_cell_2/MatMul_1MatMullstm_1/zeros:output:02lstm_1/lstm_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	2�2
lstm_1/lstm_cell_2/MatMul_1�
lstm_1/lstm_cell_2/addAddV2#lstm_1/lstm_cell_2/MatMul:product:0%lstm_1/lstm_cell_2/MatMul_1:product:0*
T0*
_output_shapes
:	2�2
lstm_1/lstm_cell_2/add�
)lstm_1/lstm_cell_2/BiasAdd/ReadVariableOpReadVariableOp2lstm_1_lstm_cell_2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02+
)lstm_1/lstm_cell_2/BiasAdd/ReadVariableOp�
lstm_1/lstm_cell_2/BiasAddBiasAddlstm_1/lstm_cell_2/add:z:01lstm_1/lstm_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	2�2
lstm_1/lstm_cell_2/BiasAddv
lstm_1/lstm_cell_2/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_1/lstm_cell_2/Const�
"lstm_1/lstm_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"lstm_1/lstm_cell_2/split/split_dim�
lstm_1/lstm_cell_2/splitSplit+lstm_1/lstm_cell_2/split/split_dim:output:0#lstm_1/lstm_cell_2/BiasAdd:output:0*
T0*@
_output_shapes.
,:	2�:	2�:	2�:	2�*
	num_split2
lstm_1/lstm_cell_2/split�
lstm_1/lstm_cell_2/SigmoidSigmoid!lstm_1/lstm_cell_2/split:output:0*
T0*
_output_shapes
:	2�2
lstm_1/lstm_cell_2/Sigmoid�
lstm_1/lstm_cell_2/Sigmoid_1Sigmoid!lstm_1/lstm_cell_2/split:output:1*
T0*
_output_shapes
:	2�2
lstm_1/lstm_cell_2/Sigmoid_1�
lstm_1/lstm_cell_2/mulMul lstm_1/lstm_cell_2/Sigmoid_1:y:0lstm_1/zeros_1:output:0*
T0*
_output_shapes
:	2�2
lstm_1/lstm_cell_2/mul�
lstm_1/lstm_cell_2/ReluRelu!lstm_1/lstm_cell_2/split:output:2*
T0*
_output_shapes
:	2�2
lstm_1/lstm_cell_2/Relu�
lstm_1/lstm_cell_2/mul_1Mullstm_1/lstm_cell_2/Sigmoid:y:0%lstm_1/lstm_cell_2/Relu:activations:0*
T0*
_output_shapes
:	2�2
lstm_1/lstm_cell_2/mul_1�
lstm_1/lstm_cell_2/add_1AddV2lstm_1/lstm_cell_2/mul:z:0lstm_1/lstm_cell_2/mul_1:z:0*
T0*
_output_shapes
:	2�2
lstm_1/lstm_cell_2/add_1�
lstm_1/lstm_cell_2/Sigmoid_2Sigmoid!lstm_1/lstm_cell_2/split:output:3*
T0*
_output_shapes
:	2�2
lstm_1/lstm_cell_2/Sigmoid_2�
lstm_1/lstm_cell_2/Relu_1Relulstm_1/lstm_cell_2/add_1:z:0*
T0*
_output_shapes
:	2�2
lstm_1/lstm_cell_2/Relu_1�
lstm_1/lstm_cell_2/mul_2Mul lstm_1/lstm_cell_2/Sigmoid_2:y:0'lstm_1/lstm_cell_2/Relu_1:activations:0*
T0*
_output_shapes
:	2�2
lstm_1/lstm_cell_2/mul_2�
$lstm_1/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"2   �   2&
$lstm_1/TensorArrayV2_1/element_shape�
lstm_1/TensorArrayV2_1TensorListReserve-lstm_1/TensorArrayV2_1/element_shape:output:0lstm_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_1/TensorArrayV2_1\
lstm_1/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_1/time�
lstm_1/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2!
lstm_1/while/maximum_iterationsx
lstm_1/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_1/while/loop_counter�
lstm_1/whileWhile"lstm_1/while/loop_counter:output:0(lstm_1/while/maximum_iterations:output:0lstm_1/time:output:0lstm_1/TensorArrayV2_1:handle:0lstm_1/zeros:output:0lstm_1/zeros_1:output:0lstm_1/strided_slice_1:output:0>lstm_1/TensorArrayUnstack/TensorListFromTensor:output_handle:01lstm_1_lstm_cell_2_matmul_readvariableop_resource3lstm_1_lstm_cell_2_matmul_1_readvariableop_resource2lstm_1_lstm_cell_2_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*<
_output_shapes*
(: : : : :	2�:	2�: : : : : *%
_read_only_resource_inputs
	
*%
bodyR
lstm_1_while_body_1394060*%
condR
lstm_1_while_cond_1394059*;
output_shapes*
(: : : : :	2�:	2�: : : : : *
parallel_iterations 2
lstm_1/while�
7lstm_1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"2   �   29
7lstm_1/TensorArrayV2Stack/TensorListStack/element_shape�
)lstm_1/TensorArrayV2Stack/TensorListStackTensorListStacklstm_1/while:output:3@lstm_1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*#
_output_shapes
:2�*
element_dtype02+
)lstm_1/TensorArrayV2Stack/TensorListStack�
lstm_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
lstm_1/strided_slice_3/stack�
lstm_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2 
lstm_1/strided_slice_3/stack_1�
lstm_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_1/strided_slice_3/stack_2�
lstm_1/strided_slice_3StridedSlice2lstm_1/TensorArrayV2Stack/TensorListStack:tensor:0%lstm_1/strided_slice_3/stack:output:0'lstm_1/strided_slice_3/stack_1:output:0'lstm_1/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
:	2�*
shrink_axis_mask2
lstm_1/strided_slice_3�
lstm_1/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_1/transpose_1/perm�
lstm_1/transpose_1	Transpose2lstm_1/TensorArrayV2Stack/TensorListStack:tensor:0 lstm_1/transpose_1/perm:output:0*
T0*#
_output_shapes
:2�2
lstm_1/transpose_1t
lstm_1/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_1/runtimeq
lstm_2/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"2      �   2
lstm_2/Shape�
lstm_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_2/strided_slice/stack�
lstm_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_2/strided_slice/stack_1�
lstm_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_2/strided_slice/stack_2�
lstm_2/strided_sliceStridedSlicelstm_2/Shape:output:0#lstm_2/strided_slice/stack:output:0%lstm_2/strided_slice/stack_1:output:0%lstm_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_2/strided_slicej
lstm_2/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
lstm_2/zeros/mul/y�
lstm_2/zeros/mulMullstm_2/strided_slice:output:0lstm_2/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_2/zeros/mulm
lstm_2/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
lstm_2/zeros/Less/y�
lstm_2/zeros/LessLesslstm_2/zeros/mul:z:0lstm_2/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_2/zeros/Lessp
lstm_2/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
lstm_2/zeros/packed/1�
lstm_2/zeros/packedPacklstm_2/strided_slice:output:0lstm_2/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_2/zeros/packedm
lstm_2/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_2/zeros/Const�
lstm_2/zerosFilllstm_2/zeros/packed:output:0lstm_2/zeros/Const:output:0*
T0*
_output_shapes

:2@2
lstm_2/zerosn
lstm_2/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
lstm_2/zeros_1/mul/y�
lstm_2/zeros_1/mulMullstm_2/strided_slice:output:0lstm_2/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_2/zeros_1/mulq
lstm_2/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
lstm_2/zeros_1/Less/y�
lstm_2/zeros_1/LessLesslstm_2/zeros_1/mul:z:0lstm_2/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_2/zeros_1/Lesst
lstm_2/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
lstm_2/zeros_1/packed/1�
lstm_2/zeros_1/packedPacklstm_2/strided_slice:output:0 lstm_2/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_2/zeros_1/packedq
lstm_2/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_2/zeros_1/Const�
lstm_2/zeros_1Filllstm_2/zeros_1/packed:output:0lstm_2/zeros_1/Const:output:0*
T0*
_output_shapes

:2@2
lstm_2/zeros_1�
lstm_2/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_2/transpose/perm�
lstm_2/transpose	Transposelstm_1/transpose_1:y:0lstm_2/transpose/perm:output:0*
T0*#
_output_shapes
:2�2
lstm_2/transposeu
lstm_2/Shape_1Const*
_output_shapes
:*
dtype0*!
valueB"   2   �   2
lstm_2/Shape_1�
lstm_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_2/strided_slice_1/stack�
lstm_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_2/strided_slice_1/stack_1�
lstm_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_2/strided_slice_1/stack_2�
lstm_2/strided_slice_1StridedSlicelstm_2/Shape_1:output:0%lstm_2/strided_slice_1/stack:output:0'lstm_2/strided_slice_1/stack_1:output:0'lstm_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_2/strided_slice_1�
"lstm_2/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2$
"lstm_2/TensorArrayV2/element_shape�
lstm_2/TensorArrayV2TensorListReserve+lstm_2/TensorArrayV2/element_shape:output:0lstm_2/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_2/TensorArrayV2�
<lstm_2/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"2   �   2>
<lstm_2/TensorArrayUnstack/TensorListFromTensor/element_shape�
.lstm_2/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_2/transpose:y:0Elstm_2/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type020
.lstm_2/TensorArrayUnstack/TensorListFromTensor�
lstm_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_2/strided_slice_2/stack�
lstm_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_2/strided_slice_2/stack_1�
lstm_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_2/strided_slice_2/stack_2�
lstm_2/strided_slice_2StridedSlicelstm_2/transpose:y:0%lstm_2/strided_slice_2/stack:output:0'lstm_2/strided_slice_2/stack_1:output:0'lstm_2/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	2�*
shrink_axis_mask2
lstm_2/strided_slice_2�
(lstm_2/lstm_cell_3/MatMul/ReadVariableOpReadVariableOp1lstm_2_lstm_cell_3_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02*
(lstm_2/lstm_cell_3/MatMul/ReadVariableOp�
lstm_2/lstm_cell_3/MatMulMatMullstm_2/strided_slice_2:output:00lstm_2/lstm_cell_3/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2�2
lstm_2/lstm_cell_3/MatMul�
*lstm_2/lstm_cell_3/MatMul_1/ReadVariableOpReadVariableOp3lstm_2_lstm_cell_3_matmul_1_readvariableop_resource*
_output_shapes
:	@�*
dtype02,
*lstm_2/lstm_cell_3/MatMul_1/ReadVariableOp�
lstm_2/lstm_cell_3/MatMul_1MatMullstm_2/zeros:output:02lstm_2/lstm_cell_3/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	2�2
lstm_2/lstm_cell_3/MatMul_1�
lstm_2/lstm_cell_3/addAddV2#lstm_2/lstm_cell_3/MatMul:product:0%lstm_2/lstm_cell_3/MatMul_1:product:0*
T0*
_output_shapes
:	2�2
lstm_2/lstm_cell_3/add�
)lstm_2/lstm_cell_3/BiasAdd/ReadVariableOpReadVariableOp2lstm_2_lstm_cell_3_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02+
)lstm_2/lstm_cell_3/BiasAdd/ReadVariableOp�
lstm_2/lstm_cell_3/BiasAddBiasAddlstm_2/lstm_cell_3/add:z:01lstm_2/lstm_cell_3/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	2�2
lstm_2/lstm_cell_3/BiasAddv
lstm_2/lstm_cell_3/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_2/lstm_cell_3/Const�
"lstm_2/lstm_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"lstm_2/lstm_cell_3/split/split_dim�
lstm_2/lstm_cell_3/splitSplit+lstm_2/lstm_cell_3/split/split_dim:output:0#lstm_2/lstm_cell_3/BiasAdd:output:0*
T0*<
_output_shapes*
(:2@:2@:2@:2@*
	num_split2
lstm_2/lstm_cell_3/split�
lstm_2/lstm_cell_3/SigmoidSigmoid!lstm_2/lstm_cell_3/split:output:0*
T0*
_output_shapes

:2@2
lstm_2/lstm_cell_3/Sigmoid�
lstm_2/lstm_cell_3/Sigmoid_1Sigmoid!lstm_2/lstm_cell_3/split:output:1*
T0*
_output_shapes

:2@2
lstm_2/lstm_cell_3/Sigmoid_1�
lstm_2/lstm_cell_3/mulMul lstm_2/lstm_cell_3/Sigmoid_1:y:0lstm_2/zeros_1:output:0*
T0*
_output_shapes

:2@2
lstm_2/lstm_cell_3/mul�
lstm_2/lstm_cell_3/ReluRelu!lstm_2/lstm_cell_3/split:output:2*
T0*
_output_shapes

:2@2
lstm_2/lstm_cell_3/Relu�
lstm_2/lstm_cell_3/mul_1Mullstm_2/lstm_cell_3/Sigmoid:y:0%lstm_2/lstm_cell_3/Relu:activations:0*
T0*
_output_shapes

:2@2
lstm_2/lstm_cell_3/mul_1�
lstm_2/lstm_cell_3/add_1AddV2lstm_2/lstm_cell_3/mul:z:0lstm_2/lstm_cell_3/mul_1:z:0*
T0*
_output_shapes

:2@2
lstm_2/lstm_cell_3/add_1�
lstm_2/lstm_cell_3/Sigmoid_2Sigmoid!lstm_2/lstm_cell_3/split:output:3*
T0*
_output_shapes

:2@2
lstm_2/lstm_cell_3/Sigmoid_2�
lstm_2/lstm_cell_3/Relu_1Relulstm_2/lstm_cell_3/add_1:z:0*
T0*
_output_shapes

:2@2
lstm_2/lstm_cell_3/Relu_1�
lstm_2/lstm_cell_3/mul_2Mul lstm_2/lstm_cell_3/Sigmoid_2:y:0'lstm_2/lstm_cell_3/Relu_1:activations:0*
T0*
_output_shapes

:2@2
lstm_2/lstm_cell_3/mul_2�
$lstm_2/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"2   @   2&
$lstm_2/TensorArrayV2_1/element_shape�
lstm_2/TensorArrayV2_1TensorListReserve-lstm_2/TensorArrayV2_1/element_shape:output:0lstm_2/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_2/TensorArrayV2_1\
lstm_2/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_2/time�
lstm_2/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2!
lstm_2/while/maximum_iterationsx
lstm_2/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_2/while/loop_counter�
lstm_2/whileWhile"lstm_2/while/loop_counter:output:0(lstm_2/while/maximum_iterations:output:0lstm_2/time:output:0lstm_2/TensorArrayV2_1:handle:0lstm_2/zeros:output:0lstm_2/zeros_1:output:0lstm_2/strided_slice_1:output:0>lstm_2/TensorArrayUnstack/TensorListFromTensor:output_handle:01lstm_2_lstm_cell_3_matmul_readvariableop_resource3lstm_2_lstm_cell_3_matmul_1_readvariableop_resource2lstm_2_lstm_cell_3_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*:
_output_shapes(
&: : : : :2@:2@: : : : : *%
_read_only_resource_inputs
	
*%
bodyR
lstm_2_while_body_1394209*%
condR
lstm_2_while_cond_1394208*9
output_shapes(
&: : : : :2@:2@: : : : : *
parallel_iterations 2
lstm_2/while�
7lstm_2/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"2   @   29
7lstm_2/TensorArrayV2Stack/TensorListStack/element_shape�
)lstm_2/TensorArrayV2Stack/TensorListStackTensorListStacklstm_2/while:output:3@lstm_2/TensorArrayV2Stack/TensorListStack/element_shape:output:0*"
_output_shapes
:2@*
element_dtype02+
)lstm_2/TensorArrayV2Stack/TensorListStack�
lstm_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
lstm_2/strided_slice_3/stack�
lstm_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2 
lstm_2/strided_slice_3/stack_1�
lstm_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_2/strided_slice_3/stack_2�
lstm_2/strided_slice_3StridedSlice2lstm_2/TensorArrayV2Stack/TensorListStack:tensor:0%lstm_2/strided_slice_3/stack:output:0'lstm_2/strided_slice_3/stack_1:output:0'lstm_2/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:2@*
shrink_axis_mask2
lstm_2/strided_slice_3�
lstm_2/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_2/transpose_1/perm�
lstm_2/transpose_1	Transpose2lstm_2/TensorArrayV2Stack/TensorListStack:tensor:0 lstm_2/transpose_1/perm:output:0*
T0*"
_output_shapes
:2@2
lstm_2/transpose_1t
lstm_2/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_2/runtime�
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:@d*
dtype02
dense/MatMul/ReadVariableOp�
dense/MatMulMatMullstm_2/strided_slice_3:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2d2
dense/MatMul�
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
dense/BiasAdd/ReadVariableOp�
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2d2
dense/BiasAdda

dense/ReluReludense/BiasAdd:output:0*
T0*
_output_shapes

:2d2

dense/Relu�
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:d*
dtype02
dense_1/MatMul/ReadVariableOp�
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:22
dense_1/MatMul�
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp�
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:22
dense_1/BiasAdd`
SoftmaxSoftmaxdense_1/BiasAdd:output:0*
T0*
_output_shapes

:22	
Softmax^
Cast_1Castinput_cc*

DstT0*

SrcT0*"
_output_shapes
:2\2
Cast_1s
lstm_cc/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"2      \   2
lstm_cc/Shape�
lstm_cc/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_cc/strided_slice/stack�
lstm_cc/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_cc/strided_slice/stack_1�
lstm_cc/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_cc/strided_slice/stack_2�
lstm_cc/strided_sliceStridedSlicelstm_cc/Shape:output:0$lstm_cc/strided_slice/stack:output:0&lstm_cc/strided_slice/stack_1:output:0&lstm_cc/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_cc/strided_slicel
lstm_cc/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
lstm_cc/zeros/mul/y�
lstm_cc/zeros/mulMullstm_cc/strided_slice:output:0lstm_cc/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_cc/zeros/mulo
lstm_cc/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
lstm_cc/zeros/Less/y�
lstm_cc/zeros/LessLesslstm_cc/zeros/mul:z:0lstm_cc/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_cc/zeros/Lessr
lstm_cc/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
lstm_cc/zeros/packed/1�
lstm_cc/zeros/packedPacklstm_cc/strided_slice:output:0lstm_cc/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_cc/zeros/packedo
lstm_cc/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_cc/zeros/Const�
lstm_cc/zerosFilllstm_cc/zeros/packed:output:0lstm_cc/zeros/Const:output:0*
T0*
_output_shapes

:2@2
lstm_cc/zerosp
lstm_cc/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
lstm_cc/zeros_1/mul/y�
lstm_cc/zeros_1/mulMullstm_cc/strided_slice:output:0lstm_cc/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_cc/zeros_1/muls
lstm_cc/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
lstm_cc/zeros_1/Less/y�
lstm_cc/zeros_1/LessLesslstm_cc/zeros_1/mul:z:0lstm_cc/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_cc/zeros_1/Lessv
lstm_cc/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
lstm_cc/zeros_1/packed/1�
lstm_cc/zeros_1/packedPacklstm_cc/strided_slice:output:0!lstm_cc/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_cc/zeros_1/packeds
lstm_cc/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_cc/zeros_1/Const�
lstm_cc/zeros_1Filllstm_cc/zeros_1/packed:output:0lstm_cc/zeros_1/Const:output:0*
T0*
_output_shapes

:2@2
lstm_cc/zeros_1�
lstm_cc/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_cc/transpose/perm�
lstm_cc/transpose	Transpose
Cast_1:y:0lstm_cc/transpose/perm:output:0*
T0*"
_output_shapes
:2\2
lstm_cc/transposew
lstm_cc/Shape_1Const*
_output_shapes
:*
dtype0*!
valueB"   2   \   2
lstm_cc/Shape_1�
lstm_cc/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_cc/strided_slice_1/stack�
lstm_cc/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_cc/strided_slice_1/stack_1�
lstm_cc/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_cc/strided_slice_1/stack_2�
lstm_cc/strided_slice_1StridedSlicelstm_cc/Shape_1:output:0&lstm_cc/strided_slice_1/stack:output:0(lstm_cc/strided_slice_1/stack_1:output:0(lstm_cc/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_cc/strided_slice_1�
#lstm_cc/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2%
#lstm_cc/TensorArrayV2/element_shape�
lstm_cc/TensorArrayV2TensorListReserve,lstm_cc/TensorArrayV2/element_shape:output:0 lstm_cc/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_cc/TensorArrayV2�
=lstm_cc/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"2   \   2?
=lstm_cc/TensorArrayUnstack/TensorListFromTensor/element_shape�
/lstm_cc/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_cc/transpose:y:0Flstm_cc/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_cc/TensorArrayUnstack/TensorListFromTensor�
lstm_cc/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_cc/strided_slice_2/stack�
lstm_cc/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_cc/strided_slice_2/stack_1�
lstm_cc/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_cc/strided_slice_2/stack_2�
lstm_cc/strided_slice_2StridedSlicelstm_cc/transpose:y:0&lstm_cc/strided_slice_2/stack:output:0(lstm_cc/strided_slice_2/stack_1:output:0(lstm_cc/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:2\*
shrink_axis_mask2
lstm_cc/strided_slice_2�
'lstm_cc/lstm_cell/MatMul/ReadVariableOpReadVariableOp0lstm_cc_lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	\�*
dtype02)
'lstm_cc/lstm_cell/MatMul/ReadVariableOp�
lstm_cc/lstm_cell/MatMulMatMul lstm_cc/strided_slice_2:output:0/lstm_cc/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2�2
lstm_cc/lstm_cell/MatMul�
)lstm_cc/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp2lstm_cc_lstm_cell_matmul_1_readvariableop_resource*
_output_shapes
:	@�*
dtype02+
)lstm_cc/lstm_cell/MatMul_1/ReadVariableOp�
lstm_cc/lstm_cell/MatMul_1MatMullstm_cc/zeros:output:01lstm_cc/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	2�2
lstm_cc/lstm_cell/MatMul_1�
lstm_cc/lstm_cell/addAddV2"lstm_cc/lstm_cell/MatMul:product:0$lstm_cc/lstm_cell/MatMul_1:product:0*
T0*
_output_shapes
:	2�2
lstm_cc/lstm_cell/add�
(lstm_cc/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp1lstm_cc_lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02*
(lstm_cc/lstm_cell/BiasAdd/ReadVariableOp�
lstm_cc/lstm_cell/BiasAddBiasAddlstm_cc/lstm_cell/add:z:00lstm_cc/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	2�2
lstm_cc/lstm_cell/BiasAddt
lstm_cc/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cc/lstm_cell/Const�
!lstm_cc/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!lstm_cc/lstm_cell/split/split_dim�
lstm_cc/lstm_cell/splitSplit*lstm_cc/lstm_cell/split/split_dim:output:0"lstm_cc/lstm_cell/BiasAdd:output:0*
T0*<
_output_shapes*
(:2@:2@:2@:2@*
	num_split2
lstm_cc/lstm_cell/split�
lstm_cc/lstm_cell/SigmoidSigmoid lstm_cc/lstm_cell/split:output:0*
T0*
_output_shapes

:2@2
lstm_cc/lstm_cell/Sigmoid�
lstm_cc/lstm_cell/Sigmoid_1Sigmoid lstm_cc/lstm_cell/split:output:1*
T0*
_output_shapes

:2@2
lstm_cc/lstm_cell/Sigmoid_1�
lstm_cc/lstm_cell/mulMullstm_cc/lstm_cell/Sigmoid_1:y:0lstm_cc/zeros_1:output:0*
T0*
_output_shapes

:2@2
lstm_cc/lstm_cell/mul�
lstm_cc/lstm_cell/ReluRelu lstm_cc/lstm_cell/split:output:2*
T0*
_output_shapes

:2@2
lstm_cc/lstm_cell/Relu�
lstm_cc/lstm_cell/mul_1Mullstm_cc/lstm_cell/Sigmoid:y:0$lstm_cc/lstm_cell/Relu:activations:0*
T0*
_output_shapes

:2@2
lstm_cc/lstm_cell/mul_1�
lstm_cc/lstm_cell/add_1AddV2lstm_cc/lstm_cell/mul:z:0lstm_cc/lstm_cell/mul_1:z:0*
T0*
_output_shapes

:2@2
lstm_cc/lstm_cell/add_1�
lstm_cc/lstm_cell/Sigmoid_2Sigmoid lstm_cc/lstm_cell/split:output:3*
T0*
_output_shapes

:2@2
lstm_cc/lstm_cell/Sigmoid_2�
lstm_cc/lstm_cell/Relu_1Relulstm_cc/lstm_cell/add_1:z:0*
T0*
_output_shapes

:2@2
lstm_cc/lstm_cell/Relu_1�
lstm_cc/lstm_cell/mul_2Mullstm_cc/lstm_cell/Sigmoid_2:y:0&lstm_cc/lstm_cell/Relu_1:activations:0*
T0*
_output_shapes

:2@2
lstm_cc/lstm_cell/mul_2�
%lstm_cc/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"2   @   2'
%lstm_cc/TensorArrayV2_1/element_shape�
lstm_cc/TensorArrayV2_1TensorListReserve.lstm_cc/TensorArrayV2_1/element_shape:output:0 lstm_cc/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_cc/TensorArrayV2_1^
lstm_cc/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_cc/time�
 lstm_cc/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2"
 lstm_cc/while/maximum_iterationsz
lstm_cc/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_cc/while/loop_counter�
lstm_cc/whileWhile#lstm_cc/while/loop_counter:output:0)lstm_cc/while/maximum_iterations:output:0lstm_cc/time:output:0 lstm_cc/TensorArrayV2_1:handle:0lstm_cc/zeros:output:0lstm_cc/zeros_1:output:0 lstm_cc/strided_slice_1:output:0?lstm_cc/TensorArrayUnstack/TensorListFromTensor:output_handle:00lstm_cc_lstm_cell_matmul_readvariableop_resource2lstm_cc_lstm_cell_matmul_1_readvariableop_resource1lstm_cc_lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*:
_output_shapes(
&: : : : :2@:2@: : : : : *%
_read_only_resource_inputs
	
*&
bodyR
lstm_cc_while_body_1394373*&
condR
lstm_cc_while_cond_1394372*9
output_shapes(
&: : : : :2@:2@: : : : : *
parallel_iterations 2
lstm_cc/while�
8lstm_cc/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"2   @   2:
8lstm_cc/TensorArrayV2Stack/TensorListStack/element_shape�
*lstm_cc/TensorArrayV2Stack/TensorListStackTensorListStacklstm_cc/while:output:3Alstm_cc/TensorArrayV2Stack/TensorListStack/element_shape:output:0*"
_output_shapes
:2@*
element_dtype02,
*lstm_cc/TensorArrayV2Stack/TensorListStack�
lstm_cc/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
lstm_cc/strided_slice_3/stack�
lstm_cc/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_cc/strided_slice_3/stack_1�
lstm_cc/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_cc/strided_slice_3/stack_2�
lstm_cc/strided_slice_3StridedSlice3lstm_cc/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_cc/strided_slice_3/stack:output:0(lstm_cc/strided_slice_3/stack_1:output:0(lstm_cc/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:2@*
shrink_axis_mask2
lstm_cc/strided_slice_3�
lstm_cc/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_cc/transpose_1/perm�
lstm_cc/transpose_1	Transpose3lstm_cc/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_cc/transpose_1/perm:output:0*
T0*"
_output_shapes
:2@2
lstm_cc/transpose_1v
lstm_cc/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_cc/runtime�
dense_cc/MatMul/ReadVariableOpReadVariableOp'dense_cc_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02 
dense_cc/MatMul/ReadVariableOp�
dense_cc/MatMulMatMul lstm_cc/strided_slice_3:output:0&dense_cc/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:22
dense_cc/MatMul�
dense_cc/BiasAdd/ReadVariableOpReadVariableOp(dense_cc_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_cc/BiasAdd/ReadVariableOp�
dense_cc/BiasAddBiasAdddense_cc/MatMul:product:0'dense_cc/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:22
dense_cc/BiasAdde
	Softmax_1Softmaxdense_cc/BiasAdd:output:0*
T0*
_output_shapes

:22
	Softmax_1�
IdentityIdentitySoftmax:softmax:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp ^dense_cc/BiasAdd/ReadVariableOp^dense_cc/MatMul/ReadVariableOp(^lstm/lstm_cell_1/BiasAdd/ReadVariableOp'^lstm/lstm_cell_1/MatMul/ReadVariableOp)^lstm/lstm_cell_1/MatMul_1/ReadVariableOp^lstm/while*^lstm_1/lstm_cell_2/BiasAdd/ReadVariableOp)^lstm_1/lstm_cell_2/MatMul/ReadVariableOp+^lstm_1/lstm_cell_2/MatMul_1/ReadVariableOp^lstm_1/while*^lstm_2/lstm_cell_3/BiasAdd/ReadVariableOp)^lstm_2/lstm_cell_3/MatMul/ReadVariableOp+^lstm_2/lstm_cell_3/MatMul_1/ReadVariableOp^lstm_2/while)^lstm_cc/lstm_cell/BiasAdd/ReadVariableOp(^lstm_cc/lstm_cell/MatMul/ReadVariableOp*^lstm_cc/lstm_cell/MatMul_1/ReadVariableOp^lstm_cc/while*
T0*
_output_shapes

:22

Identity�

Identity_1IdentitySoftmax_1:softmax:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp ^dense_cc/BiasAdd/ReadVariableOp^dense_cc/MatMul/ReadVariableOp(^lstm/lstm_cell_1/BiasAdd/ReadVariableOp'^lstm/lstm_cell_1/MatMul/ReadVariableOp)^lstm/lstm_cell_1/MatMul_1/ReadVariableOp^lstm/while*^lstm_1/lstm_cell_2/BiasAdd/ReadVariableOp)^lstm_1/lstm_cell_2/MatMul/ReadVariableOp+^lstm_1/lstm_cell_2/MatMul_1/ReadVariableOp^lstm_1/while*^lstm_2/lstm_cell_3/BiasAdd/ReadVariableOp)^lstm_2/lstm_cell_3/MatMul/ReadVariableOp+^lstm_2/lstm_cell_3/MatMul_1/ReadVariableOp^lstm_2/while)^lstm_cc/lstm_cell/BiasAdd/ReadVariableOp(^lstm_cc/lstm_cell/MatMul/ReadVariableOp*^lstm_cc/lstm_cell/MatMul_1/ReadVariableOp^lstm_cc/while*
T0*
_output_shapes

:22

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*w
_input_shapesf
d:2\:2\::::::::::::::::::2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2B
dense_cc/BiasAdd/ReadVariableOpdense_cc/BiasAdd/ReadVariableOp2@
dense_cc/MatMul/ReadVariableOpdense_cc/MatMul/ReadVariableOp2R
'lstm/lstm_cell_1/BiasAdd/ReadVariableOp'lstm/lstm_cell_1/BiasAdd/ReadVariableOp2P
&lstm/lstm_cell_1/MatMul/ReadVariableOp&lstm/lstm_cell_1/MatMul/ReadVariableOp2T
(lstm/lstm_cell_1/MatMul_1/ReadVariableOp(lstm/lstm_cell_1/MatMul_1/ReadVariableOp2

lstm/while
lstm/while2V
)lstm_1/lstm_cell_2/BiasAdd/ReadVariableOp)lstm_1/lstm_cell_2/BiasAdd/ReadVariableOp2T
(lstm_1/lstm_cell_2/MatMul/ReadVariableOp(lstm_1/lstm_cell_2/MatMul/ReadVariableOp2X
*lstm_1/lstm_cell_2/MatMul_1/ReadVariableOp*lstm_1/lstm_cell_2/MatMul_1/ReadVariableOp2
lstm_1/whilelstm_1/while2V
)lstm_2/lstm_cell_3/BiasAdd/ReadVariableOp)lstm_2/lstm_cell_3/BiasAdd/ReadVariableOp2T
(lstm_2/lstm_cell_3/MatMul/ReadVariableOp(lstm_2/lstm_cell_3/MatMul/ReadVariableOp2X
*lstm_2/lstm_cell_3/MatMul_1/ReadVariableOp*lstm_2/lstm_cell_3/MatMul_1/ReadVariableOp2
lstm_2/whilelstm_2/while2T
(lstm_cc/lstm_cell/BiasAdd/ReadVariableOp(lstm_cc/lstm_cell/BiasAdd/ReadVariableOp2R
'lstm_cc/lstm_cell/MatMul/ReadVariableOp'lstm_cc/lstm_cell/MatMul/ReadVariableOp2V
)lstm_cc/lstm_cell/MatMul_1/ReadVariableOp)lstm_cc/lstm_cell/MatMul_1/ReadVariableOp2
lstm_cc/whilelstm_cc/while:L H
"
_output_shapes
:2\
"
_user_specified_name
input_ac:LH
"
_output_shapes
:2\
"
_user_specified_name
input_cc
�B
�
while_body_1395518
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_06
2while_lstm_cell_3_matmul_readvariableop_resource_08
4while_lstm_cell_3_matmul_1_readvariableop_resource_07
3while_lstm_cell_3_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor4
0while_lstm_cell_3_matmul_readvariableop_resource6
2while_lstm_cell_3_matmul_1_readvariableop_resource5
1while_lstm_cell_3_biasadd_readvariableop_resource��(while/lstm_cell_3/BiasAdd/ReadVariableOp�'while/lstm_cell_3/MatMul/ReadVariableOp�)while/lstm_cell_3/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�����   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape�
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:����������*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem�
'while/lstm_cell_3/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_3_matmul_readvariableop_resource_0* 
_output_shapes
:
��*
dtype02)
'while/lstm_cell_3/MatMul/ReadVariableOp�
while/lstm_cell_3/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_3/MatMul�
)while/lstm_cell_3/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_3_matmul_1_readvariableop_resource_0*
_output_shapes
:	@�*
dtype02+
)while/lstm_cell_3/MatMul_1/ReadVariableOp�
while/lstm_cell_3/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_3/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_3/MatMul_1�
while/lstm_cell_3/addAddV2"while/lstm_cell_3/MatMul:product:0$while/lstm_cell_3/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_3/add�
(while/lstm_cell_3/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_3_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02*
(while/lstm_cell_3/BiasAdd/ReadVariableOp�
while/lstm_cell_3/BiasAddBiasAddwhile/lstm_cell_3/add:z:00while/lstm_cell_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_3/BiasAddt
while/lstm_cell_3/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_3/Const�
!while/lstm_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_3/split/split_dim�
while/lstm_cell_3/splitSplit*while/lstm_cell_3/split/split_dim:output:0"while/lstm_cell_3/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������@:���������@:���������@:���������@*
	num_split2
while/lstm_cell_3/split�
while/lstm_cell_3/SigmoidSigmoid while/lstm_cell_3/split:output:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_3/Sigmoid�
while/lstm_cell_3/Sigmoid_1Sigmoid while/lstm_cell_3/split:output:1*
T0*'
_output_shapes
:���������@2
while/lstm_cell_3/Sigmoid_1�
while/lstm_cell_3/mulMulwhile/lstm_cell_3/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������@2
while/lstm_cell_3/mul�
while/lstm_cell_3/ReluRelu while/lstm_cell_3/split:output:2*
T0*'
_output_shapes
:���������@2
while/lstm_cell_3/Relu�
while/lstm_cell_3/mul_1Mulwhile/lstm_cell_3/Sigmoid:y:0$while/lstm_cell_3/Relu:activations:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_3/mul_1�
while/lstm_cell_3/add_1AddV2while/lstm_cell_3/mul:z:0while/lstm_cell_3/mul_1:z:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_3/add_1�
while/lstm_cell_3/Sigmoid_2Sigmoid while/lstm_cell_3/split:output:3*
T0*'
_output_shapes
:���������@2
while/lstm_cell_3/Sigmoid_2�
while/lstm_cell_3/Relu_1Reluwhile/lstm_cell_3/add_1:z:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_3/Relu_1�
while/lstm_cell_3/mul_2Mulwhile/lstm_cell_3/Sigmoid_2:y:0&while/lstm_cell_3/Relu_1:activations:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_3/mul_2�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_3/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1�
while/IdentityIdentitywhile/add_1:z:0)^while/lstm_cell_3/BiasAdd/ReadVariableOp(^while/lstm_cell_3/MatMul/ReadVariableOp*^while/lstm_cell_3/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity�
while/Identity_1Identitywhile_while_maximum_iterations)^while/lstm_cell_3/BiasAdd/ReadVariableOp(^while/lstm_cell_3/MatMul/ReadVariableOp*^while/lstm_cell_3/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1�
while/Identity_2Identitywhile/add:z:0)^while/lstm_cell_3/BiasAdd/ReadVariableOp(^while/lstm_cell_3/MatMul/ReadVariableOp*^while/lstm_cell_3/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2�
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0)^while/lstm_cell_3/BiasAdd/ReadVariableOp(^while/lstm_cell_3/MatMul/ReadVariableOp*^while/lstm_cell_3/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3�
while/Identity_4Identitywhile/lstm_cell_3/mul_2:z:0)^while/lstm_cell_3/BiasAdd/ReadVariableOp(^while/lstm_cell_3/MatMul/ReadVariableOp*^while/lstm_cell_3/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:���������@2
while/Identity_4�
while/Identity_5Identitywhile/lstm_cell_3/add_1:z:0)^while/lstm_cell_3/BiasAdd/ReadVariableOp(^while/lstm_cell_3/MatMul/ReadVariableOp*^while/lstm_cell_3/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:���������@2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_3_biasadd_readvariableop_resource3while_lstm_cell_3_biasadd_readvariableop_resource_0"j
2while_lstm_cell_3_matmul_1_readvariableop_resource4while_lstm_cell_3_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_3_matmul_readvariableop_resource2while_lstm_cell_3_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :���������@:���������@: : :::2T
(while/lstm_cell_3/BiasAdd/ReadVariableOp(while/lstm_cell_3/BiasAdd/ReadVariableOp2R
'while/lstm_cell_3/MatMul/ReadVariableOp'while/lstm_cell_3/MatMul/ReadVariableOp2V
)while/lstm_cell_3/MatMul_1/ReadVariableOp)while/lstm_cell_3/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������@:-)
'
_output_shapes
:���������@:

_output_shapes
: :

_output_shapes
: 
�
�
while_cond_1395342
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1395342___redundant_placeholder05
1while_while_cond_1395342___redundant_placeholder15
1while_while_cond_1395342___redundant_placeholder25
1while_while_cond_1395342___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*U
_input_shapesD
B: : : : :����������:����������: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:����������:.*
(
_output_shapes
:����������:

_output_shapes
: :

_output_shapes
:
�
�
-__inference_lstm_cell_1_layer_call_fn_1395978

inputs
states_0
states_1
unknown
	unknown_0
	unknown_1
identity

identity_1

identity_2��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������@:���������@:���������@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_lstm_cell_1_layer_call_and_return_conditional_losses_13921172
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������@2

Identity�

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:���������@2

Identity_1�

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:���������@2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:���������\:���������@:���������@:::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������\
 
_user_specified_nameinputs:QM
'
_output_shapes
:���������@
"
_user_specified_name
states/0:QM
'
_output_shapes
:���������@
"
_user_specified_name
states/1
�/
�
 __inference__traced_save_1396253
file_prefix.
*savev2_dense_cc_kernel_read_readvariableop,
(savev2_dense_cc_bias_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop7
3savev2_lstm_cc_lstm_cell_kernel_read_readvariableopA
=savev2_lstm_cc_lstm_cell_recurrent_kernel_read_readvariableop5
1savev2_lstm_cc_lstm_cell_bias_read_readvariableop6
2savev2_lstm_lstm_cell_1_kernel_read_readvariableop@
<savev2_lstm_lstm_cell_1_recurrent_kernel_read_readvariableop4
0savev2_lstm_lstm_cell_1_bias_read_readvariableop8
4savev2_lstm_1_lstm_cell_2_kernel_read_readvariableopB
>savev2_lstm_1_lstm_cell_2_recurrent_kernel_read_readvariableop6
2savev2_lstm_1_lstm_cell_2_bias_read_readvariableop8
4savev2_lstm_2_lstm_cell_3_kernel_read_readvariableopB
>savev2_lstm_2_lstm_cell_3_recurrent_kernel_read_readvariableop6
2savev2_lstm_2_lstm_cell_3_bias_read_readvariableop
savev2_const

identity_1��MergeV2Checkpoints�
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard�
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B*dense_cc/kernel/.ATTRIBUTES/VARIABLE_VALUEB(dense_cc/bias/.ATTRIBUTES/VARIABLE_VALUEB*dens1_ac/kernel/.ATTRIBUTES/VARIABLE_VALUEB(dens1_ac/bias/.ATTRIBUTES/VARIABLE_VALUEB+dense2_ac/kernel/.ATTRIBUTES/VARIABLE_VALUEB)dense2_ac/bias/.ATTRIBUTES/VARIABLE_VALUEB.lstm_cc/cell/kernel/.ATTRIBUTES/VARIABLE_VALUEB8lstm_cc/cell/recurrent_kernel/.ATTRIBUTES/VARIABLE_VALUEB,lstm_cc/cell/bias/.ATTRIBUTES/VARIABLE_VALUEB/lstm1_ac/cell/kernel/.ATTRIBUTES/VARIABLE_VALUEB9lstm1_ac/cell/recurrent_kernel/.ATTRIBUTES/VARIABLE_VALUEB-lstm1_ac/cell/bias/.ATTRIBUTES/VARIABLE_VALUEB/lstm2_ac/cell/kernel/.ATTRIBUTES/VARIABLE_VALUEB9lstm2_ac/cell/recurrent_kernel/.ATTRIBUTES/VARIABLE_VALUEB-lstm2_ac/cell/bias/.ATTRIBUTES/VARIABLE_VALUEB/lstm3_ac/cell/kernel/.ATTRIBUTES/VARIABLE_VALUEB9lstm3_ac/cell/recurrent_kernel/.ATTRIBUTES/VARIABLE_VALUEB-lstm3_ac/cell/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*9
value0B.B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_cc_kernel_read_readvariableop(savev2_dense_cc_bias_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop3savev2_lstm_cc_lstm_cell_kernel_read_readvariableop=savev2_lstm_cc_lstm_cell_recurrent_kernel_read_readvariableop1savev2_lstm_cc_lstm_cell_bias_read_readvariableop2savev2_lstm_lstm_cell_1_kernel_read_readvariableop<savev2_lstm_lstm_cell_1_recurrent_kernel_read_readvariableop0savev2_lstm_lstm_cell_1_bias_read_readvariableop4savev2_lstm_1_lstm_cell_2_kernel_read_readvariableop>savev2_lstm_1_lstm_cell_2_recurrent_kernel_read_readvariableop2savev2_lstm_1_lstm_cell_2_bias_read_readvariableop4savev2_lstm_2_lstm_cell_3_kernel_read_readvariableop>savev2_lstm_2_lstm_cell_3_recurrent_kernel_read_readvariableop2savev2_lstm_2_lstm_cell_3_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *!
dtypes
22
SaveV2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*�
_input_shapes�
�: :@::@d:d:d::	\�:	@�:�:	\�:	@�:�:	@�:
��:�:
��:	@�:�: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:@: 

_output_shapes
::$ 

_output_shapes

:@d: 

_output_shapes
:d:$ 

_output_shapes

:d: 

_output_shapes
::%!

_output_shapes
:	\�:%!

_output_shapes
:	@�:!	

_output_shapes	
:�:%
!

_output_shapes
:	\�:%!

_output_shapes
:	@�:!

_output_shapes	
:�:%!

_output_shapes
:	@�:&"
 
_output_shapes
:
��:!

_output_shapes	
:�:&"
 
_output_shapes
:
��:%!

_output_shapes
:	@�:!

_output_shapes	
:�:

_output_shapes
: 
�%
�
while_body_1393631
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_lstm_cell_3_1393655_0
while_lstm_cell_3_1393657_0
while_lstm_cell_3_1393659_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_lstm_cell_3_1393655
while_lstm_cell_3_1393657
while_lstm_cell_3_1393659��)while/lstm_cell_3/StatefulPartitionedCall�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�����   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape�
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:����������*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem�
)while/lstm_cell_3/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_3_1393655_0while_lstm_cell_3_1393657_0while_lstm_cell_3_1393659_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������@:���������@:���������@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_lstm_cell_3_layer_call_and_return_conditional_losses_13933042+
)while/lstm_cell_3/StatefulPartitionedCall�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/lstm_cell_3/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1�
while/IdentityIdentitywhile/add_1:z:0*^while/lstm_cell_3/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity�
while/Identity_1Identitywhile_while_maximum_iterations*^while/lstm_cell_3/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1�
while/Identity_2Identitywhile/add:z:0*^while/lstm_cell_3/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2�
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*^while/lstm_cell_3/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3�
while/Identity_4Identity2while/lstm_cell_3/StatefulPartitionedCall:output:1*^while/lstm_cell_3/StatefulPartitionedCall*
T0*'
_output_shapes
:���������@2
while/Identity_4�
while/Identity_5Identity2while/lstm_cell_3/StatefulPartitionedCall:output:2*^while/lstm_cell_3/StatefulPartitionedCall*
T0*'
_output_shapes
:���������@2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"8
while_lstm_cell_3_1393655while_lstm_cell_3_1393655_0"8
while_lstm_cell_3_1393657while_lstm_cell_3_1393657_0"8
while_lstm_cell_3_1393659while_lstm_cell_3_1393659_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :���������@:���������@: : :::2V
)while/lstm_cell_3/StatefulPartitionedCall)while/lstm_cell_3/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������@:-)
'
_output_shapes
:���������@:

_output_shapes
: :

_output_shapes
: 
�Z
�
C__inference_lstm_2_layer_call_and_return_conditional_losses_1395603
inputs_0.
*lstm_cell_3_matmul_readvariableop_resource0
,lstm_cell_3_matmul_1_readvariableop_resource/
+lstm_cell_3_biasadd_readvariableop_resource
identity��"lstm_cell_3/BiasAdd/ReadVariableOp�!lstm_cell_3/MatMul/ReadVariableOp�#lstm_cell_3/MatMul_1/ReadVariableOp�whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros/packed/1�
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������@2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/packed/1�
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:���������@2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm�
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*5
_output_shapes#
!:�������������������2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1�
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2
TensorArrayV2/element_shape�
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2�
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�����   27
5TensorArrayUnstack/TensorListFromTensor/element_shape�
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*
shrink_axis_mask2
strided_slice_2�
!lstm_cell_3/MatMul/ReadVariableOpReadVariableOp*lstm_cell_3_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02#
!lstm_cell_3/MatMul/ReadVariableOp�
lstm_cell_3/MatMulMatMulstrided_slice_2:output:0)lstm_cell_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_3/MatMul�
#lstm_cell_3/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_3_matmul_1_readvariableop_resource*
_output_shapes
:	@�*
dtype02%
#lstm_cell_3/MatMul_1/ReadVariableOp�
lstm_cell_3/MatMul_1MatMulzeros:output:0+lstm_cell_3/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_3/MatMul_1�
lstm_cell_3/addAddV2lstm_cell_3/MatMul:product:0lstm_cell_3/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
lstm_cell_3/add�
"lstm_cell_3/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_3_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02$
"lstm_cell_3/BiasAdd/ReadVariableOp�
lstm_cell_3/BiasAddBiasAddlstm_cell_3/add:z:0*lstm_cell_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_3/BiasAddh
lstm_cell_3/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_3/Const|
lstm_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_3/split/split_dim�
lstm_cell_3/splitSplit$lstm_cell_3/split/split_dim:output:0lstm_cell_3/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������@:���������@:���������@:���������@*
	num_split2
lstm_cell_3/split�
lstm_cell_3/SigmoidSigmoidlstm_cell_3/split:output:0*
T0*'
_output_shapes
:���������@2
lstm_cell_3/Sigmoid�
lstm_cell_3/Sigmoid_1Sigmoidlstm_cell_3/split:output:1*
T0*'
_output_shapes
:���������@2
lstm_cell_3/Sigmoid_1�
lstm_cell_3/mulMullstm_cell_3/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������@2
lstm_cell_3/mulz
lstm_cell_3/ReluRelulstm_cell_3/split:output:2*
T0*'
_output_shapes
:���������@2
lstm_cell_3/Relu�
lstm_cell_3/mul_1Mullstm_cell_3/Sigmoid:y:0lstm_cell_3/Relu:activations:0*
T0*'
_output_shapes
:���������@2
lstm_cell_3/mul_1�
lstm_cell_3/add_1AddV2lstm_cell_3/mul:z:0lstm_cell_3/mul_1:z:0*
T0*'
_output_shapes
:���������@2
lstm_cell_3/add_1�
lstm_cell_3/Sigmoid_2Sigmoidlstm_cell_3/split:output:3*
T0*'
_output_shapes
:���������@2
lstm_cell_3/Sigmoid_2y
lstm_cell_3/Relu_1Relulstm_cell_3/add_1:z:0*
T0*'
_output_shapes
:���������@2
lstm_cell_3/Relu_1�
lstm_cell_3/mul_2Mullstm_cell_3/Sigmoid_2:y:0 lstm_cell_3/Relu_1:activations:0*
T0*'
_output_shapes
:���������@2
lstm_cell_3/mul_2�
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   2
TensorArrayV2_1/element_shape�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter�
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_3_matmul_readvariableop_resource,lstm_cell_3_matmul_1_readvariableop_resource+lstm_cell_3_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������@:���������@: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_1395518*
condR
while_cond_1395517*K
output_shapes:
8: : : : :���������@:���������@: : : : : *
parallel_iterations 2
while�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   22
0TensorArrayV2Stack/TensorListStack/element_shape�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������@*
element_dtype02$
"TensorArrayV2Stack/TensorListStack�
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������@*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������@2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime�
IdentityIdentitystrided_slice_3:output:0#^lstm_cell_3/BiasAdd/ReadVariableOp"^lstm_cell_3/MatMul/ReadVariableOp$^lstm_cell_3/MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:�������������������:::2H
"lstm_cell_3/BiasAdd/ReadVariableOp"lstm_cell_3/BiasAdd/ReadVariableOp2F
!lstm_cell_3/MatMul/ReadVariableOp!lstm_cell_3/MatMul/ReadVariableOp2J
#lstm_cell_3/MatMul_1/ReadVariableOp#lstm_cell_3/MatMul_1/ReadVariableOp2
whilewhile:_ [
5
_output_shapes#
!:�������������������
"
_user_specified_name
inputs/0
�
�
H__inference_lstm_cell_3_layer_call_and_return_conditional_losses_1396144

inputs
states_0
states_1"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1

identity_2��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	@�*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:����������2
add�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:���������@:���������@:���������@:���������@*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:���������@2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:���������@2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:���������@2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:���������@2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:���������@2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:���������@2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:���������@2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:���������@2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:���������@2
mul_2�
IdentityIdentity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:���������@2

Identity�

Identity_1Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:���������@2

Identity_1�

Identity_2Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:���������@2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*Y
_input_shapesH
F:����������:���������@:���������@:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:QM
'
_output_shapes
:���������@
"
_user_specified_name
states/0:QM
'
_output_shapes
:���������@
"
_user_specified_name
states/1
�B
�
while_body_1395343
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_06
2while_lstm_cell_2_matmul_readvariableop_resource_08
4while_lstm_cell_2_matmul_1_readvariableop_resource_07
3while_lstm_cell_2_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor4
0while_lstm_cell_2_matmul_readvariableop_resource6
2while_lstm_cell_2_matmul_1_readvariableop_resource5
1while_lstm_cell_2_biasadd_readvariableop_resource��(while/lstm_cell_2/BiasAdd/ReadVariableOp�'while/lstm_cell_2/MatMul/ReadVariableOp�)while/lstm_cell_2/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape�
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������@*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem�
'while/lstm_cell_2/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_2_matmul_readvariableop_resource_0*
_output_shapes
:	@�*
dtype02)
'while/lstm_cell_2/MatMul/ReadVariableOp�
while/lstm_cell_2/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_2/MatMul�
)while/lstm_cell_2/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_2_matmul_1_readvariableop_resource_0* 
_output_shapes
:
��*
dtype02+
)while/lstm_cell_2/MatMul_1/ReadVariableOp�
while/lstm_cell_2/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_2/MatMul_1�
while/lstm_cell_2/addAddV2"while/lstm_cell_2/MatMul:product:0$while/lstm_cell_2/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_2/add�
(while/lstm_cell_2/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_2_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02*
(while/lstm_cell_2/BiasAdd/ReadVariableOp�
while/lstm_cell_2/BiasAddBiasAddwhile/lstm_cell_2/add:z:00while/lstm_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_2/BiasAddt
while/lstm_cell_2/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_2/Const�
!while/lstm_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_2/split/split_dim�
while/lstm_cell_2/splitSplit*while/lstm_cell_2/split/split_dim:output:0"while/lstm_cell_2/BiasAdd:output:0*
T0*d
_output_shapesR
P:����������:����������:����������:����������*
	num_split2
while/lstm_cell_2/split�
while/lstm_cell_2/SigmoidSigmoid while/lstm_cell_2/split:output:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_2/Sigmoid�
while/lstm_cell_2/Sigmoid_1Sigmoid while/lstm_cell_2/split:output:1*
T0*(
_output_shapes
:����������2
while/lstm_cell_2/Sigmoid_1�
while/lstm_cell_2/mulMulwhile/lstm_cell_2/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:����������2
while/lstm_cell_2/mul�
while/lstm_cell_2/ReluRelu while/lstm_cell_2/split:output:2*
T0*(
_output_shapes
:����������2
while/lstm_cell_2/Relu�
while/lstm_cell_2/mul_1Mulwhile/lstm_cell_2/Sigmoid:y:0$while/lstm_cell_2/Relu:activations:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_2/mul_1�
while/lstm_cell_2/add_1AddV2while/lstm_cell_2/mul:z:0while/lstm_cell_2/mul_1:z:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_2/add_1�
while/lstm_cell_2/Sigmoid_2Sigmoid while/lstm_cell_2/split:output:3*
T0*(
_output_shapes
:����������2
while/lstm_cell_2/Sigmoid_2�
while/lstm_cell_2/Relu_1Reluwhile/lstm_cell_2/add_1:z:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_2/Relu_1�
while/lstm_cell_2/mul_2Mulwhile/lstm_cell_2/Sigmoid_2:y:0&while/lstm_cell_2/Relu_1:activations:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_2/mul_2�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_2/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1�
while/IdentityIdentitywhile/add_1:z:0)^while/lstm_cell_2/BiasAdd/ReadVariableOp(^while/lstm_cell_2/MatMul/ReadVariableOp*^while/lstm_cell_2/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity�
while/Identity_1Identitywhile_while_maximum_iterations)^while/lstm_cell_2/BiasAdd/ReadVariableOp(^while/lstm_cell_2/MatMul/ReadVariableOp*^while/lstm_cell_2/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1�
while/Identity_2Identitywhile/add:z:0)^while/lstm_cell_2/BiasAdd/ReadVariableOp(^while/lstm_cell_2/MatMul/ReadVariableOp*^while/lstm_cell_2/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2�
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0)^while/lstm_cell_2/BiasAdd/ReadVariableOp(^while/lstm_cell_2/MatMul/ReadVariableOp*^while/lstm_cell_2/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3�
while/Identity_4Identitywhile/lstm_cell_2/mul_2:z:0)^while/lstm_cell_2/BiasAdd/ReadVariableOp(^while/lstm_cell_2/MatMul/ReadVariableOp*^while/lstm_cell_2/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:����������2
while/Identity_4�
while/Identity_5Identitywhile/lstm_cell_2/add_1:z:0)^while/lstm_cell_2/BiasAdd/ReadVariableOp(^while/lstm_cell_2/MatMul/ReadVariableOp*^while/lstm_cell_2/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:����������2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_2_biasadd_readvariableop_resource3while_lstm_cell_2_biasadd_readvariableop_resource_0"j
2while_lstm_cell_2_matmul_1_readvariableop_resource4while_lstm_cell_2_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_2_matmul_readvariableop_resource2while_lstm_cell_2_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :����������:����������: : :::2T
(while/lstm_cell_2/BiasAdd/ReadVariableOp(while/lstm_cell_2/BiasAdd/ReadVariableOp2R
'while/lstm_cell_2/MatMul/ReadVariableOp'while/lstm_cell_2/MatMul/ReadVariableOp2V
)while/lstm_cell_2/MatMul_1/ReadVariableOp)while/lstm_cell_2/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:����������:.*
(
_output_shapes
:����������:

_output_shapes
: :

_output_shapes
: 
�N
�

#__inference__traced_restore_1396317
file_prefix$
 assignvariableop_dense_cc_kernel$
 assignvariableop_1_dense_cc_bias#
assignvariableop_2_dense_kernel!
assignvariableop_3_dense_bias%
!assignvariableop_4_dense_1_kernel#
assignvariableop_5_dense_1_bias/
+assignvariableop_6_lstm_cc_lstm_cell_kernel9
5assignvariableop_7_lstm_cc_lstm_cell_recurrent_kernel-
)assignvariableop_8_lstm_cc_lstm_cell_bias.
*assignvariableop_9_lstm_lstm_cell_1_kernel9
5assignvariableop_10_lstm_lstm_cell_1_recurrent_kernel-
)assignvariableop_11_lstm_lstm_cell_1_bias1
-assignvariableop_12_lstm_1_lstm_cell_2_kernel;
7assignvariableop_13_lstm_1_lstm_cell_2_recurrent_kernel/
+assignvariableop_14_lstm_1_lstm_cell_2_bias1
-assignvariableop_15_lstm_2_lstm_cell_3_kernel;
7assignvariableop_16_lstm_2_lstm_cell_3_recurrent_kernel/
+assignvariableop_17_lstm_2_lstm_cell_3_bias
identity_19��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_2�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B*dense_cc/kernel/.ATTRIBUTES/VARIABLE_VALUEB(dense_cc/bias/.ATTRIBUTES/VARIABLE_VALUEB*dens1_ac/kernel/.ATTRIBUTES/VARIABLE_VALUEB(dens1_ac/bias/.ATTRIBUTES/VARIABLE_VALUEB+dense2_ac/kernel/.ATTRIBUTES/VARIABLE_VALUEB)dense2_ac/bias/.ATTRIBUTES/VARIABLE_VALUEB.lstm_cc/cell/kernel/.ATTRIBUTES/VARIABLE_VALUEB8lstm_cc/cell/recurrent_kernel/.ATTRIBUTES/VARIABLE_VALUEB,lstm_cc/cell/bias/.ATTRIBUTES/VARIABLE_VALUEB/lstm1_ac/cell/kernel/.ATTRIBUTES/VARIABLE_VALUEB9lstm1_ac/cell/recurrent_kernel/.ATTRIBUTES/VARIABLE_VALUEB-lstm1_ac/cell/bias/.ATTRIBUTES/VARIABLE_VALUEB/lstm2_ac/cell/kernel/.ATTRIBUTES/VARIABLE_VALUEB9lstm2_ac/cell/recurrent_kernel/.ATTRIBUTES/VARIABLE_VALUEB-lstm2_ac/cell/bias/.ATTRIBUTES/VARIABLE_VALUEB/lstm3_ac/cell/kernel/.ATTRIBUTES/VARIABLE_VALUEB9lstm3_ac/cell/recurrent_kernel/.ATTRIBUTES/VARIABLE_VALUEB-lstm3_ac/cell/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*9
value0B.B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*`
_output_shapesN
L:::::::::::::::::::*!
dtypes
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOp assignvariableop_dense_cc_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_cc_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOpassignvariableop_2_dense_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOp!assignvariableop_4_dense_1_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense_1_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOp+assignvariableop_6_lstm_cc_lstm_cell_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOp5assignvariableop_7_lstm_cc_lstm_cell_recurrent_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOp)assignvariableop_8_lstm_cc_lstm_cell_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOp*assignvariableop_9_lstm_lstm_cell_1_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10�
AssignVariableOp_10AssignVariableOp5assignvariableop_10_lstm_lstm_cell_1_recurrent_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11�
AssignVariableOp_11AssignVariableOp)assignvariableop_11_lstm_lstm_cell_1_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12�
AssignVariableOp_12AssignVariableOp-assignvariableop_12_lstm_1_lstm_cell_2_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13�
AssignVariableOp_13AssignVariableOp7assignvariableop_13_lstm_1_lstm_cell_2_recurrent_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14�
AssignVariableOp_14AssignVariableOp+assignvariableop_14_lstm_1_lstm_cell_2_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15�
AssignVariableOp_15AssignVariableOp-assignvariableop_15_lstm_2_lstm_cell_3_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16�
AssignVariableOp_16AssignVariableOp7assignvariableop_16_lstm_2_lstm_cell_3_recurrent_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17�
AssignVariableOp_17AssignVariableOp+assignvariableop_17_lstm_2_lstm_cell_3_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_179
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�
Identity_18Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_18�
Identity_19IdentityIdentity_18:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_19"#
identity_19Identity_19:output:0*]
_input_shapesL
J: ::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
F__inference_lstm_cell_layer_call_and_return_conditional_losses_1391507

inputs

states
states_1"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1

identity_2��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	\�*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	@�*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:����������2
add�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:���������@:���������@:���������@:���������@*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:���������@2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:���������@2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:���������@2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:���������@2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:���������@2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:���������@2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:���������@2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:���������@2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:���������@2
mul_2�
IdentityIdentity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:���������@2

Identity�

Identity_1Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:���������@2

Identity_1�

Identity_2Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:���������@2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:���������\:���������@:���������@:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:���������\
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������@
 
_user_specified_namestates:OK
'
_output_shapes
:���������@
 
_user_specified_namestates
�%
�
while_body_1393763
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_lstm_cell_3_1393787_0
while_lstm_cell_3_1393789_0
while_lstm_cell_3_1393791_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_lstm_cell_3_1393787
while_lstm_cell_3_1393789
while_lstm_cell_3_1393791��)while/lstm_cell_3/StatefulPartitionedCall�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�����   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape�
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:����������*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem�
)while/lstm_cell_3/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_3_1393787_0while_lstm_cell_3_1393789_0while_lstm_cell_3_1393791_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������@:���������@:���������@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_lstm_cell_3_layer_call_and_return_conditional_losses_13933372+
)while/lstm_cell_3/StatefulPartitionedCall�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/lstm_cell_3/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1�
while/IdentityIdentitywhile/add_1:z:0*^while/lstm_cell_3/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity�
while/Identity_1Identitywhile_while_maximum_iterations*^while/lstm_cell_3/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1�
while/Identity_2Identitywhile/add:z:0*^while/lstm_cell_3/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2�
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*^while/lstm_cell_3/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3�
while/Identity_4Identity2while/lstm_cell_3/StatefulPartitionedCall:output:1*^while/lstm_cell_3/StatefulPartitionedCall*
T0*'
_output_shapes
:���������@2
while/Identity_4�
while/Identity_5Identity2while/lstm_cell_3/StatefulPartitionedCall:output:2*^while/lstm_cell_3/StatefulPartitionedCall*
T0*'
_output_shapes
:���������@2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"8
while_lstm_cell_3_1393787while_lstm_cell_3_1393787_0"8
while_lstm_cell_3_1393789while_lstm_cell_3_1393789_0"8
while_lstm_cell_3_1393791while_lstm_cell_3_1393791_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :���������@:���������@: : :::2V
)while/lstm_cell_3/StatefulPartitionedCall)while/lstm_cell_3/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������@:-)
'
_output_shapes
:���������@:

_output_shapes
: :

_output_shapes
: 
�D
�
C__inference_lstm_2_layer_call_and_return_conditional_losses_1393700

inputs
lstm_cell_3_1393618
lstm_cell_3_1393620
lstm_cell_3_1393622
identity��#lstm_cell_3/StatefulPartitionedCall�whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros/packed/1�
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������@2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/packed/1�
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:���������@2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm�
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:�������������������2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1�
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2
TensorArrayV2/element_shape�
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2�
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�����   27
5TensorArrayUnstack/TensorListFromTensor/element_shape�
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*
shrink_axis_mask2
strided_slice_2�
#lstm_cell_3/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_3_1393618lstm_cell_3_1393620lstm_cell_3_1393622*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������@:���������@:���������@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_lstm_cell_3_layer_call_and_return_conditional_losses_13933042%
#lstm_cell_3/StatefulPartitionedCall�
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   2
TensorArrayV2_1/element_shape�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter�
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_3_1393618lstm_cell_3_1393620lstm_cell_3_1393622*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������@:���������@: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_1393631*
condR
while_cond_1393630*K
output_shapes:
8: : : : :���������@:���������@: : : : : *
parallel_iterations 2
while�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   22
0TensorArrayV2Stack/TensorListStack/element_shape�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������@*
element_dtype02$
"TensorArrayV2Stack/TensorListStack�
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������@*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������@2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime�
IdentityIdentitystrided_slice_3:output:0$^lstm_cell_3/StatefulPartitionedCall^while*
T0*'
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:�������������������:::2J
#lstm_cell_3/StatefulPartitionedCall#lstm_cell_3/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:�������������������
 
_user_specified_nameinputs
�	
�
lstm_while_cond_1393910&
"lstm_while_lstm_while_loop_counter,
(lstm_while_lstm_while_maximum_iterations
lstm_while_placeholder
lstm_while_placeholder_1
lstm_while_placeholder_2
lstm_while_placeholder_3(
$lstm_while_less_lstm_strided_slice_1?
;lstm_while_lstm_while_cond_1393910___redundant_placeholder0?
;lstm_while_lstm_while_cond_1393910___redundant_placeholder1?
;lstm_while_lstm_while_cond_1393910___redundant_placeholder2?
;lstm_while_lstm_while_cond_1393910___redundant_placeholder3
lstm_while_identity
�
lstm/while/LessLesslstm_while_placeholder$lstm_while_less_lstm_strided_slice_1*
T0*
_output_shapes
: 2
lstm/while/Lessl
lstm/while/IdentityIdentitylstm/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm/while/Identity"3
lstm_while_identitylstm/while/Identity:output:0*A
_input_shapes0
.: : : : :2@:2@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:2@:$ 

_output_shapes

:2@:

_output_shapes
: :

_output_shapes
:
�	
�
lstm_1_while_cond_1394059*
&lstm_1_while_lstm_1_while_loop_counter0
,lstm_1_while_lstm_1_while_maximum_iterations
lstm_1_while_placeholder
lstm_1_while_placeholder_1
lstm_1_while_placeholder_2
lstm_1_while_placeholder_3,
(lstm_1_while_less_lstm_1_strided_slice_1C
?lstm_1_while_lstm_1_while_cond_1394059___redundant_placeholder0C
?lstm_1_while_lstm_1_while_cond_1394059___redundant_placeholder1C
?lstm_1_while_lstm_1_while_cond_1394059___redundant_placeholder2C
?lstm_1_while_lstm_1_while_cond_1394059___redundant_placeholder3
lstm_1_while_identity
�
lstm_1/while/LessLesslstm_1_while_placeholder(lstm_1_while_less_lstm_1_strided_slice_1*
T0*
_output_shapes
: 2
lstm_1/while/Lessr
lstm_1/while/IdentityIdentitylstm_1/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_1/while/Identity"7
lstm_1_while_identitylstm_1/while/Identity:output:0*C
_input_shapes2
0: : : : :	2�:	2�: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	2�:%!

_output_shapes
:	2�:

_output_shapes
: :

_output_shapes
:
�B
�
while_body_1394862
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_06
2while_lstm_cell_1_matmul_readvariableop_resource_08
4while_lstm_cell_1_matmul_1_readvariableop_resource_07
3while_lstm_cell_1_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor4
0while_lstm_cell_1_matmul_readvariableop_resource6
2while_lstm_cell_1_matmul_1_readvariableop_resource5
1while_lstm_cell_1_biasadd_readvariableop_resource��(while/lstm_cell_1/BiasAdd/ReadVariableOp�'while/lstm_cell_1/MatMul/ReadVariableOp�)while/lstm_cell_1/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����\   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape�
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������\*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem�
'while/lstm_cell_1/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_1_matmul_readvariableop_resource_0*
_output_shapes
:	\�*
dtype02)
'while/lstm_cell_1/MatMul/ReadVariableOp�
while/lstm_cell_1/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_1/MatMul�
)while/lstm_cell_1/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_1_matmul_1_readvariableop_resource_0*
_output_shapes
:	@�*
dtype02+
)while/lstm_cell_1/MatMul_1/ReadVariableOp�
while/lstm_cell_1/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_1/MatMul_1�
while/lstm_cell_1/addAddV2"while/lstm_cell_1/MatMul:product:0$while/lstm_cell_1/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_1/add�
(while/lstm_cell_1/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_1_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02*
(while/lstm_cell_1/BiasAdd/ReadVariableOp�
while/lstm_cell_1/BiasAddBiasAddwhile/lstm_cell_1/add:z:00while/lstm_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_1/BiasAddt
while/lstm_cell_1/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_1/Const�
!while/lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_1/split/split_dim�
while/lstm_cell_1/splitSplit*while/lstm_cell_1/split/split_dim:output:0"while/lstm_cell_1/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������@:���������@:���������@:���������@*
	num_split2
while/lstm_cell_1/split�
while/lstm_cell_1/SigmoidSigmoid while/lstm_cell_1/split:output:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_1/Sigmoid�
while/lstm_cell_1/Sigmoid_1Sigmoid while/lstm_cell_1/split:output:1*
T0*'
_output_shapes
:���������@2
while/lstm_cell_1/Sigmoid_1�
while/lstm_cell_1/mulMulwhile/lstm_cell_1/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������@2
while/lstm_cell_1/mul�
while/lstm_cell_1/ReluRelu while/lstm_cell_1/split:output:2*
T0*'
_output_shapes
:���������@2
while/lstm_cell_1/Relu�
while/lstm_cell_1/mul_1Mulwhile/lstm_cell_1/Sigmoid:y:0$while/lstm_cell_1/Relu:activations:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_1/mul_1�
while/lstm_cell_1/add_1AddV2while/lstm_cell_1/mul:z:0while/lstm_cell_1/mul_1:z:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_1/add_1�
while/lstm_cell_1/Sigmoid_2Sigmoid while/lstm_cell_1/split:output:3*
T0*'
_output_shapes
:���������@2
while/lstm_cell_1/Sigmoid_2�
while/lstm_cell_1/Relu_1Reluwhile/lstm_cell_1/add_1:z:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_1/Relu_1�
while/lstm_cell_1/mul_2Mulwhile/lstm_cell_1/Sigmoid_2:y:0&while/lstm_cell_1/Relu_1:activations:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_1/mul_2�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_1/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1�
while/IdentityIdentitywhile/add_1:z:0)^while/lstm_cell_1/BiasAdd/ReadVariableOp(^while/lstm_cell_1/MatMul/ReadVariableOp*^while/lstm_cell_1/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity�
while/Identity_1Identitywhile_while_maximum_iterations)^while/lstm_cell_1/BiasAdd/ReadVariableOp(^while/lstm_cell_1/MatMul/ReadVariableOp*^while/lstm_cell_1/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1�
while/Identity_2Identitywhile/add:z:0)^while/lstm_cell_1/BiasAdd/ReadVariableOp(^while/lstm_cell_1/MatMul/ReadVariableOp*^while/lstm_cell_1/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2�
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0)^while/lstm_cell_1/BiasAdd/ReadVariableOp(^while/lstm_cell_1/MatMul/ReadVariableOp*^while/lstm_cell_1/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3�
while/Identity_4Identitywhile/lstm_cell_1/mul_2:z:0)^while/lstm_cell_1/BiasAdd/ReadVariableOp(^while/lstm_cell_1/MatMul/ReadVariableOp*^while/lstm_cell_1/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:���������@2
while/Identity_4�
while/Identity_5Identitywhile/lstm_cell_1/add_1:z:0)^while/lstm_cell_1/BiasAdd/ReadVariableOp(^while/lstm_cell_1/MatMul/ReadVariableOp*^while/lstm_cell_1/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:���������@2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_1_biasadd_readvariableop_resource3while_lstm_cell_1_biasadd_readvariableop_resource_0"j
2while_lstm_cell_1_matmul_1_readvariableop_resource4while_lstm_cell_1_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_1_matmul_readvariableop_resource2while_lstm_cell_1_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :���������@:���������@: : :::2T
(while/lstm_cell_1/BiasAdd/ReadVariableOp(while/lstm_cell_1/BiasAdd/ReadVariableOp2R
'while/lstm_cell_1/MatMul/ReadVariableOp'while/lstm_cell_1/MatMul/ReadVariableOp2V
)while/lstm_cell_1/MatMul_1/ReadVariableOp)while/lstm_cell_1/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������@:-)
'
_output_shapes
:���������@:

_output_shapes
: :

_output_shapes
: 
�Z
�
C__inference_lstm_2_layer_call_and_return_conditional_losses_1395756
inputs_0.
*lstm_cell_3_matmul_readvariableop_resource0
,lstm_cell_3_matmul_1_readvariableop_resource/
+lstm_cell_3_biasadd_readvariableop_resource
identity��"lstm_cell_3/BiasAdd/ReadVariableOp�!lstm_cell_3/MatMul/ReadVariableOp�#lstm_cell_3/MatMul_1/ReadVariableOp�whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros/packed/1�
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������@2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/packed/1�
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:���������@2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm�
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*5
_output_shapes#
!:�������������������2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1�
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2
TensorArrayV2/element_shape�
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2�
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�����   27
5TensorArrayUnstack/TensorListFromTensor/element_shape�
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*
shrink_axis_mask2
strided_slice_2�
!lstm_cell_3/MatMul/ReadVariableOpReadVariableOp*lstm_cell_3_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02#
!lstm_cell_3/MatMul/ReadVariableOp�
lstm_cell_3/MatMulMatMulstrided_slice_2:output:0)lstm_cell_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_3/MatMul�
#lstm_cell_3/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_3_matmul_1_readvariableop_resource*
_output_shapes
:	@�*
dtype02%
#lstm_cell_3/MatMul_1/ReadVariableOp�
lstm_cell_3/MatMul_1MatMulzeros:output:0+lstm_cell_3/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_3/MatMul_1�
lstm_cell_3/addAddV2lstm_cell_3/MatMul:product:0lstm_cell_3/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
lstm_cell_3/add�
"lstm_cell_3/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_3_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02$
"lstm_cell_3/BiasAdd/ReadVariableOp�
lstm_cell_3/BiasAddBiasAddlstm_cell_3/add:z:0*lstm_cell_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_3/BiasAddh
lstm_cell_3/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_3/Const|
lstm_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_3/split/split_dim�
lstm_cell_3/splitSplit$lstm_cell_3/split/split_dim:output:0lstm_cell_3/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������@:���������@:���������@:���������@*
	num_split2
lstm_cell_3/split�
lstm_cell_3/SigmoidSigmoidlstm_cell_3/split:output:0*
T0*'
_output_shapes
:���������@2
lstm_cell_3/Sigmoid�
lstm_cell_3/Sigmoid_1Sigmoidlstm_cell_3/split:output:1*
T0*'
_output_shapes
:���������@2
lstm_cell_3/Sigmoid_1�
lstm_cell_3/mulMullstm_cell_3/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������@2
lstm_cell_3/mulz
lstm_cell_3/ReluRelulstm_cell_3/split:output:2*
T0*'
_output_shapes
:���������@2
lstm_cell_3/Relu�
lstm_cell_3/mul_1Mullstm_cell_3/Sigmoid:y:0lstm_cell_3/Relu:activations:0*
T0*'
_output_shapes
:���������@2
lstm_cell_3/mul_1�
lstm_cell_3/add_1AddV2lstm_cell_3/mul:z:0lstm_cell_3/mul_1:z:0*
T0*'
_output_shapes
:���������@2
lstm_cell_3/add_1�
lstm_cell_3/Sigmoid_2Sigmoidlstm_cell_3/split:output:3*
T0*'
_output_shapes
:���������@2
lstm_cell_3/Sigmoid_2y
lstm_cell_3/Relu_1Relulstm_cell_3/add_1:z:0*
T0*'
_output_shapes
:���������@2
lstm_cell_3/Relu_1�
lstm_cell_3/mul_2Mullstm_cell_3/Sigmoid_2:y:0 lstm_cell_3/Relu_1:activations:0*
T0*'
_output_shapes
:���������@2
lstm_cell_3/mul_2�
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   2
TensorArrayV2_1/element_shape�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter�
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_3_matmul_readvariableop_resource,lstm_cell_3_matmul_1_readvariableop_resource+lstm_cell_3_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������@:���������@: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_1395671*
condR
while_cond_1395670*K
output_shapes:
8: : : : :���������@:���������@: : : : : *
parallel_iterations 2
while�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   22
0TensorArrayV2Stack/TensorListStack/element_shape�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������@*
element_dtype02$
"TensorArrayV2Stack/TensorListStack�
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������@*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������@2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime�
IdentityIdentitystrided_slice_3:output:0#^lstm_cell_3/BiasAdd/ReadVariableOp"^lstm_cell_3/MatMul/ReadVariableOp$^lstm_cell_3/MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:�������������������:::2H
"lstm_cell_3/BiasAdd/ReadVariableOp"lstm_cell_3/BiasAdd/ReadVariableOp2F
!lstm_cell_3/MatMul/ReadVariableOp!lstm_cell_3/MatMul/ReadVariableOp2J
#lstm_cell_3/MatMul_1/ReadVariableOp#lstm_cell_3/MatMul_1/ReadVariableOp2
whilewhile:_ [
5
_output_shapes#
!:�������������������
"
_user_specified_name
inputs/0
�
�
&__inference_lstm_layer_call_fn_1395122
inputs_0
unknown
	unknown_0
	unknown_1
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_lstm_layer_call_and_return_conditional_losses_13926122
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :������������������@2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:������������������\:::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :������������������\
"
_user_specified_name
inputs/0
�Y
�
D__inference_lstm_cc_layer_call_and_return_conditional_losses_1394772
inputs_0,
(lstm_cell_matmul_readvariableop_resource.
*lstm_cell_matmul_1_readvariableop_resource-
)lstm_cell_biasadd_readvariableop_resource
identity�� lstm_cell/BiasAdd/ReadVariableOp�lstm_cell/MatMul/ReadVariableOp�!lstm_cell/MatMul_1/ReadVariableOp�whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros/packed/1�
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������@2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/packed/1�
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:���������@2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm�
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :������������������\2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1�
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2
TensorArrayV2/element_shape�
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2�
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����\   27
5TensorArrayUnstack/TensorListFromTensor/element_shape�
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������\*
shrink_axis_mask2
strided_slice_2�
lstm_cell/MatMul/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	\�*
dtype02!
lstm_cell/MatMul/ReadVariableOp�
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell/MatMul�
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource*
_output_shapes
:	@�*
dtype02#
!lstm_cell/MatMul_1/ReadVariableOp�
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell/MatMul_1�
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
lstm_cell/add�
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp)lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02"
 lstm_cell/BiasAdd/ReadVariableOp�
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell/BiasAddd
lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Constx
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/split/split_dim�
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������@:���������@:���������@:���������@*
	num_split2
lstm_cell/split}
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*'
_output_shapes
:���������@2
lstm_cell/Sigmoid�
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*'
_output_shapes
:���������@2
lstm_cell/Sigmoid_1�
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������@2
lstm_cell/mult
lstm_cell/ReluRelulstm_cell/split:output:2*
T0*'
_output_shapes
:���������@2
lstm_cell/Relu�
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:���������@2
lstm_cell/mul_1�
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:���������@2
lstm_cell/add_1�
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*'
_output_shapes
:���������@2
lstm_cell/Sigmoid_2s
lstm_cell/Relu_1Relulstm_cell/add_1:z:0*
T0*'
_output_shapes
:���������@2
lstm_cell/Relu_1�
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:���������@2
lstm_cell/mul_2�
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   2
TensorArrayV2_1/element_shape�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter�
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0(lstm_cell_matmul_readvariableop_resource*lstm_cell_matmul_1_readvariableop_resource)lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������@:���������@: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_1394687*
condR
while_cond_1394686*K
output_shapes:
8: : : : :���������@:���������@: : : : : *
parallel_iterations 2
while�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   22
0TensorArrayV2Stack/TensorListStack/element_shape�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������@*
element_dtype02$
"TensorArrayV2Stack/TensorListStack�
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������@*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������@2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime�
IdentityIdentitystrided_slice_3:output:0!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:������������������\:::2D
 lstm_cell/BiasAdd/ReadVariableOp lstm_cell/BiasAdd/ReadVariableOp2B
lstm_cell/MatMul/ReadVariableOplstm_cell/MatMul/ReadVariableOp2F
!lstm_cell/MatMul_1/ReadVariableOp!lstm_cell/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :������������������\
"
_user_specified_name
inputs/0"�J
saver_filename:0StatefulPartitionedCall:0StatefulPartitionedCall_18"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp:��
�
lstm_cc
dense_cc
lstm1_ac
lstm2_ac
lstm3_ac
dens1_ac
	dense2_ac
	keras_api
	
signatures
	�call"�
_tf_keras_model�{"class_name": "Classifiers", "name": "classifiers", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Classifiers"}}
�

cell

state_spec
regularization_losses
trainable_variables
	variables
	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_rnn_layer�
{"class_name": "LSTM", "name": "lstm_cc", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null, 5, 132]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "lstm_cc", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null, 5, 132]}, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 64, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 92]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [50, 5, 92]}}
�

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_cc", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_cc", "trainable": true, "dtype": "float32", "units": 4, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [50, 64]}}
�
cell

state_spec
regularization_losses
trainable_variables
	variables
	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_rnn_layer�
{"class_name": "LSTM", "name": "lstm", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null, 3, 132]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "lstm", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null, 3, 132]}, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 64, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 92]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [50, 3, 92]}}
�
cell

state_spec
regularization_losses
trainable_variables
 	variables
!	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�

_tf_keras_rnn_layer�	{"class_name": "LSTM", "name": "lstm_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "lstm_1", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 128, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 64]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [50, 3, 64]}}
�
"cell
#
state_spec
$regularization_losses
%trainable_variables
&	variables
'	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�

_tf_keras_rnn_layer�	{"class_name": "LSTM", "name": "lstm_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "lstm_2", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 64, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 128]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [50, 3, 128]}}
�

(kernel
)bias
*trainable_variables
+	variables
,regularization_losses
-	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 100, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [50, 64]}}
�

.kernel
/bias
0trainable_variables
1	variables
2regularization_losses
3	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 5, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 100}}}, "build_input_shape": {"class_name": "TensorShape", "items": [50, 100]}}
"
_generic_user_object
"
signature_map
�

4kernel
5recurrent_kernel
6bias
7trainable_variables
8	variables
9regularization_losses
:	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "LSTMCell", "name": "lstm_cell", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "lstm_cell", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
40
51
62"
trackable_list_wrapper
5
40
51
62"
trackable_list_wrapper
�
;layer_metrics
regularization_losses
<non_trainable_variables
=metrics
trainable_variables

>layers
	variables
?layer_regularization_losses

@states
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
!:@2dense_cc/kernel
:2dense_cc/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Alayer_metrics
Bnon_trainable_variables
Cmetrics
trainable_variables

Dlayers
	variables
Elayer_regularization_losses
regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�

Fkernel
Grecurrent_kernel
Hbias
Itrainable_variables
J	variables
Kregularization_losses
L	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "LSTMCell", "name": "lstm_cell_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "lstm_cell_1", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
F0
G1
H2"
trackable_list_wrapper
5
F0
G1
H2"
trackable_list_wrapper
�
Mlayer_metrics
regularization_losses
Nnon_trainable_variables
Ometrics
trainable_variables

Players
	variables
Qlayer_regularization_losses

Rstates
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�

Skernel
Trecurrent_kernel
Ubias
Vtrainable_variables
W	variables
Xregularization_losses
Y	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "LSTMCell", "name": "lstm_cell_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "lstm_cell_2", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
S0
T1
U2"
trackable_list_wrapper
5
S0
T1
U2"
trackable_list_wrapper
�
Zlayer_metrics
regularization_losses
[non_trainable_variables
\metrics
trainable_variables

]layers
 	variables
^layer_regularization_losses

_states
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�

`kernel
arecurrent_kernel
bbias
ctrainable_variables
d	variables
eregularization_losses
f	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "LSTMCell", "name": "lstm_cell_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "lstm_cell_3", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
`0
a1
b2"
trackable_list_wrapper
5
`0
a1
b2"
trackable_list_wrapper
�
glayer_metrics
$regularization_losses
hnon_trainable_variables
imetrics
%trainable_variables

jlayers
&	variables
klayer_regularization_losses

lstates
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
:@d2dense/kernel
:d2
dense/bias
.
(0
)1"
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
mlayer_metrics
nnon_trainable_variables
ometrics
*trainable_variables

players
+	variables
qlayer_regularization_losses
,regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 :d2dense_1/kernel
:2dense_1/bias
.
.0
/1"
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
rlayer_metrics
snon_trainable_variables
tmetrics
0trainable_variables

ulayers
1	variables
vlayer_regularization_losses
2regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
+:)	\�2lstm_cc/lstm_cell/kernel
5:3	@�2"lstm_cc/lstm_cell/recurrent_kernel
%:#�2lstm_cc/lstm_cell/bias
5
40
51
62"
trackable_list_wrapper
5
40
51
62"
trackable_list_wrapper
 "
trackable_list_wrapper
�
wlayer_metrics
xnon_trainable_variables
ymetrics
7trainable_variables

zlayers
8	variables
{layer_regularization_losses
9regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'

0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
*:(	\�2lstm/lstm_cell_1/kernel
4:2	@�2!lstm/lstm_cell_1/recurrent_kernel
$:"�2lstm/lstm_cell_1/bias
5
F0
G1
H2"
trackable_list_wrapper
5
F0
G1
H2"
trackable_list_wrapper
 "
trackable_list_wrapper
�
|layer_metrics
}non_trainable_variables
~metrics
Itrainable_variables

layers
J	variables
 �layer_regularization_losses
Kregularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
,:*	@�2lstm_1/lstm_cell_2/kernel
7:5
��2#lstm_1/lstm_cell_2/recurrent_kernel
&:$�2lstm_1/lstm_cell_2/bias
5
S0
T1
U2"
trackable_list_wrapper
5
S0
T1
U2"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�layer_metrics
�non_trainable_variables
�metrics
Vtrainable_variables
�layers
W	variables
 �layer_regularization_losses
Xregularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
-:+
��2lstm_2/lstm_cell_3/kernel
6:4	@�2#lstm_2/lstm_cell_3/recurrent_kernel
&:$�2lstm_2/lstm_cell_3/bias
5
`0
a1
b2"
trackable_list_wrapper
5
`0
a1
b2"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�layer_metrics
�non_trainable_variables
�metrics
ctrainable_variables
�layers
d	variables
 �layer_regularization_losses
eregularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
"0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�2�
__inference_call_1394466�
���
FullArgSpec+
args#� 
jself

jinput_ac

jinput_cc
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
)__inference_lstm_cc_layer_call_fn_1394794
)__inference_lstm_cc_layer_call_fn_1394783�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
D__inference_lstm_cc_layer_call_and_return_conditional_losses_1394619
D__inference_lstm_cc_layer_call_and_return_conditional_losses_1394772�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
&__inference_lstm_layer_call_fn_1395111
&__inference_lstm_layer_call_fn_1395122�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
A__inference_lstm_layer_call_and_return_conditional_losses_1394947
A__inference_lstm_layer_call_and_return_conditional_losses_1395100�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
(__inference_lstm_1_layer_call_fn_1395439
(__inference_lstm_1_layer_call_fn_1395450�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
C__inference_lstm_1_layer_call_and_return_conditional_losses_1395428
C__inference_lstm_1_layer_call_and_return_conditional_losses_1395275�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
(__inference_lstm_2_layer_call_fn_1395767
(__inference_lstm_2_layer_call_fn_1395778�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
C__inference_lstm_2_layer_call_and_return_conditional_losses_1395603
C__inference_lstm_2_layer_call_and_return_conditional_losses_1395756�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
+__inference_lstm_cell_layer_call_fn_1395861
+__inference_lstm_cell_layer_call_fn_1395878�
���
FullArgSpec3
args+�(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
F__inference_lstm_cell_layer_call_and_return_conditional_losses_1395811
F__inference_lstm_cell_layer_call_and_return_conditional_losses_1395844�
���
FullArgSpec3
args+�(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
-__inference_lstm_cell_1_layer_call_fn_1395978
-__inference_lstm_cell_1_layer_call_fn_1395961�
���
FullArgSpec3
args+�(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
H__inference_lstm_cell_1_layer_call_and_return_conditional_losses_1395911
H__inference_lstm_cell_1_layer_call_and_return_conditional_losses_1395944�
���
FullArgSpec3
args+�(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
-__inference_lstm_cell_2_layer_call_fn_1396078
-__inference_lstm_cell_2_layer_call_fn_1396061�
���
FullArgSpec3
args+�(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
H__inference_lstm_cell_2_layer_call_and_return_conditional_losses_1396044
H__inference_lstm_cell_2_layer_call_and_return_conditional_losses_1396011�
���
FullArgSpec3
args+�(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
-__inference_lstm_cell_3_layer_call_fn_1396161
-__inference_lstm_cell_3_layer_call_fn_1396178�
���
FullArgSpec3
args+�(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
H__inference_lstm_cell_3_layer_call_and_return_conditional_losses_1396111
H__inference_lstm_cell_3_layer_call_and_return_conditional_losses_1396144�
���
FullArgSpec3
args+�(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 �
__inference_call_1394466�FGHSTU`ab()./456K�H
A�>
�
input_ac2\
�
input_cc2\
� "+�(
�
02
�
12�
C__inference_lstm_1_layer_call_and_return_conditional_losses_1395275�STUO�L
E�B
4�1
/�,
inputs/0������������������@

 
p

 
� "3�0
)�&
0�������������������
� �
C__inference_lstm_1_layer_call_and_return_conditional_losses_1395428�STUO�L
E�B
4�1
/�,
inputs/0������������������@

 
p 

 
� "3�0
)�&
0�������������������
� �
(__inference_lstm_1_layer_call_fn_1395439~STUO�L
E�B
4�1
/�,
inputs/0������������������@

 
p

 
� "&�#��������������������
(__inference_lstm_1_layer_call_fn_1395450~STUO�L
E�B
4�1
/�,
inputs/0������������������@

 
p 

 
� "&�#��������������������
C__inference_lstm_2_layer_call_and_return_conditional_losses_1395603~`abP�M
F�C
5�2
0�-
inputs/0�������������������

 
p

 
� "%�"
�
0���������@
� �
C__inference_lstm_2_layer_call_and_return_conditional_losses_1395756~`abP�M
F�C
5�2
0�-
inputs/0�������������������

 
p 

 
� "%�"
�
0���������@
� �
(__inference_lstm_2_layer_call_fn_1395767q`abP�M
F�C
5�2
0�-
inputs/0�������������������

 
p

 
� "����������@�
(__inference_lstm_2_layer_call_fn_1395778q`abP�M
F�C
5�2
0�-
inputs/0�������������������

 
p 

 
� "����������@�
D__inference_lstm_cc_layer_call_and_return_conditional_losses_1394619}456O�L
E�B
4�1
/�,
inputs/0������������������\

 
p

 
� "%�"
�
0���������@
� �
D__inference_lstm_cc_layer_call_and_return_conditional_losses_1394772}456O�L
E�B
4�1
/�,
inputs/0������������������\

 
p 

 
� "%�"
�
0���������@
� �
)__inference_lstm_cc_layer_call_fn_1394783p456O�L
E�B
4�1
/�,
inputs/0������������������\

 
p

 
� "����������@�
)__inference_lstm_cc_layer_call_fn_1394794p456O�L
E�B
4�1
/�,
inputs/0������������������\

 
p 

 
� "����������@�
H__inference_lstm_cell_1_layer_call_and_return_conditional_losses_1395911�FGH��}
v�s
 �
inputs���������\
K�H
"�
states/0���������@
"�
states/1���������@
p
� "s�p
i�f
�
0/0���������@
E�B
�
0/1/0���������@
�
0/1/1���������@
� �
H__inference_lstm_cell_1_layer_call_and_return_conditional_losses_1395944�FGH��}
v�s
 �
inputs���������\
K�H
"�
states/0���������@
"�
states/1���������@
p 
� "s�p
i�f
�
0/0���������@
E�B
�
0/1/0���������@
�
0/1/1���������@
� �
-__inference_lstm_cell_1_layer_call_fn_1395961�FGH��}
v�s
 �
inputs���������\
K�H
"�
states/0���������@
"�
states/1���������@
p
� "c�`
�
0���������@
A�>
�
1/0���������@
�
1/1���������@�
-__inference_lstm_cell_1_layer_call_fn_1395978�FGH��}
v�s
 �
inputs���������\
K�H
"�
states/0���������@
"�
states/1���������@
p 
� "c�`
�
0���������@
A�>
�
1/0���������@
�
1/1���������@�
H__inference_lstm_cell_2_layer_call_and_return_conditional_losses_1396011�STU��
x�u
 �
inputs���������@
M�J
#� 
states/0����������
#� 
states/1����������
p
� "v�s
l�i
�
0/0����������
G�D
 �
0/1/0����������
 �
0/1/1����������
� �
H__inference_lstm_cell_2_layer_call_and_return_conditional_losses_1396044�STU��
x�u
 �
inputs���������@
M�J
#� 
states/0����������
#� 
states/1����������
p 
� "v�s
l�i
�
0/0����������
G�D
 �
0/1/0����������
 �
0/1/1����������
� �
-__inference_lstm_cell_2_layer_call_fn_1396061�STU��
x�u
 �
inputs���������@
M�J
#� 
states/0����������
#� 
states/1����������
p
� "f�c
�
0����������
C�@
�
1/0����������
�
1/1�����������
-__inference_lstm_cell_2_layer_call_fn_1396078�STU��
x�u
 �
inputs���������@
M�J
#� 
states/0����������
#� 
states/1����������
p 
� "f�c
�
0����������
C�@
�
1/0����������
�
1/1�����������
H__inference_lstm_cell_3_layer_call_and_return_conditional_losses_1396111�`ab��~
w�t
!�
inputs����������
K�H
"�
states/0���������@
"�
states/1���������@
p
� "s�p
i�f
�
0/0���������@
E�B
�
0/1/0���������@
�
0/1/1���������@
� �
H__inference_lstm_cell_3_layer_call_and_return_conditional_losses_1396144�`ab��~
w�t
!�
inputs����������
K�H
"�
states/0���������@
"�
states/1���������@
p 
� "s�p
i�f
�
0/0���������@
E�B
�
0/1/0���������@
�
0/1/1���������@
� �
-__inference_lstm_cell_3_layer_call_fn_1396161�`ab��~
w�t
!�
inputs����������
K�H
"�
states/0���������@
"�
states/1���������@
p
� "c�`
�
0���������@
A�>
�
1/0���������@
�
1/1���������@�
-__inference_lstm_cell_3_layer_call_fn_1396178�`ab��~
w�t
!�
inputs����������
K�H
"�
states/0���������@
"�
states/1���������@
p 
� "c�`
�
0���������@
A�>
�
1/0���������@
�
1/1���������@�
F__inference_lstm_cell_layer_call_and_return_conditional_losses_1395811�456��}
v�s
 �
inputs���������\
K�H
"�
states/0���������@
"�
states/1���������@
p
� "s�p
i�f
�
0/0���������@
E�B
�
0/1/0���������@
�
0/1/1���������@
� �
F__inference_lstm_cell_layer_call_and_return_conditional_losses_1395844�456��}
v�s
 �
inputs���������\
K�H
"�
states/0���������@
"�
states/1���������@
p 
� "s�p
i�f
�
0/0���������@
E�B
�
0/1/0���������@
�
0/1/1���������@
� �
+__inference_lstm_cell_layer_call_fn_1395861�456��}
v�s
 �
inputs���������\
K�H
"�
states/0���������@
"�
states/1���������@
p
� "c�`
�
0���������@
A�>
�
1/0���������@
�
1/1���������@�
+__inference_lstm_cell_layer_call_fn_1395878�456��}
v�s
 �
inputs���������\
K�H
"�
states/0���������@
"�
states/1���������@
p 
� "c�`
�
0���������@
A�>
�
1/0���������@
�
1/1���������@�
A__inference_lstm_layer_call_and_return_conditional_losses_1394947�FGHO�L
E�B
4�1
/�,
inputs/0������������������\

 
p

 
� "2�/
(�%
0������������������@
� �
A__inference_lstm_layer_call_and_return_conditional_losses_1395100�FGHO�L
E�B
4�1
/�,
inputs/0������������������\

 
p 

 
� "2�/
(�%
0������������������@
� �
&__inference_lstm_layer_call_fn_1395111}FGHO�L
E�B
4�1
/�,
inputs/0������������������\

 
p

 
� "%�"������������������@�
&__inference_lstm_layer_call_fn_1395122}FGHO�L
E�B
4�1
/�,
inputs/0������������������\

 
p 

 
� "%�"������������������@