??

??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
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
9
	IdentityN

input2T
output2T"
T
list(type)(0
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
?
Mul
x"T
y"T
z"T"
Ttype:
2	?
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
dtypetype?
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
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
0
Sigmoid
x"T
y"T"
Ttype:

2
?
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
executor_typestring ?
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
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718́
n
	h0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_name	h0/kernel
g
h0/kernel/Read/ReadVariableOpReadVariableOp	h0/kernel*
_output_shapes

:*
dtype0
f
h0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name	h0/bias
_
h0/bias/Read/ReadVariableOpReadVariableOph0/bias*
_output_shapes
:*
dtype0
n
	h1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_name	h1/kernel
g
h1/kernel/Read/ReadVariableOpReadVariableOp	h1/kernel*
_output_shapes

:*
dtype0
f
h1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name	h1/bias
_
h1/bias/Read/ReadVariableOpReadVariableOph1/bias*
_output_shapes
:*
dtype0
n
	h2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_name	h2/kernel
g
h2/kernel/Read/ReadVariableOpReadVariableOp	h2/kernel*
_output_shapes

:*
dtype0
f
h2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name	h2/bias
_
h2/bias/Read/ReadVariableOpReadVariableOph2/bias*
_output_shapes
:*
dtype0
n
	h3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_name	h3/kernel
g
h3/kernel/Read/ReadVariableOpReadVariableOp	h3/kernel*
_output_shapes

:*
dtype0
f
h3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name	h3/bias
_
h3/bias/Read/ReadVariableOpReadVariableOph3/bias*
_output_shapes
:*
dtype0
v
output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_nameoutput/kernel
o
!output/kernel/Read/ReadVariableOpReadVariableOpoutput/kernel*
_output_shapes

:*
dtype0
n
output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameoutput/bias
g
output/bias/Read/ReadVariableOpReadVariableOpoutput/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
|
Adam/h0/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*!
shared_nameAdam/h0/kernel/m
u
$Adam/h0/kernel/m/Read/ReadVariableOpReadVariableOpAdam/h0/kernel/m*
_output_shapes

:*
dtype0
t
Adam/h0/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameAdam/h0/bias/m
m
"Adam/h0/bias/m/Read/ReadVariableOpReadVariableOpAdam/h0/bias/m*
_output_shapes
:*
dtype0
|
Adam/h1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*!
shared_nameAdam/h1/kernel/m
u
$Adam/h1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/h1/kernel/m*
_output_shapes

:*
dtype0
t
Adam/h1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameAdam/h1/bias/m
m
"Adam/h1/bias/m/Read/ReadVariableOpReadVariableOpAdam/h1/bias/m*
_output_shapes
:*
dtype0
|
Adam/h2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*!
shared_nameAdam/h2/kernel/m
u
$Adam/h2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/h2/kernel/m*
_output_shapes

:*
dtype0
t
Adam/h2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameAdam/h2/bias/m
m
"Adam/h2/bias/m/Read/ReadVariableOpReadVariableOpAdam/h2/bias/m*
_output_shapes
:*
dtype0
|
Adam/h3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*!
shared_nameAdam/h3/kernel/m
u
$Adam/h3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/h3/kernel/m*
_output_shapes

:*
dtype0
t
Adam/h3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameAdam/h3/bias/m
m
"Adam/h3/bias/m/Read/ReadVariableOpReadVariableOpAdam/h3/bias/m*
_output_shapes
:*
dtype0
?
Adam/output/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*%
shared_nameAdam/output/kernel/m
}
(Adam/output/kernel/m/Read/ReadVariableOpReadVariableOpAdam/output/kernel/m*
_output_shapes

:*
dtype0
|
Adam/output/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/output/bias/m
u
&Adam/output/bias/m/Read/ReadVariableOpReadVariableOpAdam/output/bias/m*
_output_shapes
:*
dtype0
|
Adam/h0/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*!
shared_nameAdam/h0/kernel/v
u
$Adam/h0/kernel/v/Read/ReadVariableOpReadVariableOpAdam/h0/kernel/v*
_output_shapes

:*
dtype0
t
Adam/h0/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameAdam/h0/bias/v
m
"Adam/h0/bias/v/Read/ReadVariableOpReadVariableOpAdam/h0/bias/v*
_output_shapes
:*
dtype0
|
Adam/h1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*!
shared_nameAdam/h1/kernel/v
u
$Adam/h1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/h1/kernel/v*
_output_shapes

:*
dtype0
t
Adam/h1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameAdam/h1/bias/v
m
"Adam/h1/bias/v/Read/ReadVariableOpReadVariableOpAdam/h1/bias/v*
_output_shapes
:*
dtype0
|
Adam/h2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*!
shared_nameAdam/h2/kernel/v
u
$Adam/h2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/h2/kernel/v*
_output_shapes

:*
dtype0
t
Adam/h2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameAdam/h2/bias/v
m
"Adam/h2/bias/v/Read/ReadVariableOpReadVariableOpAdam/h2/bias/v*
_output_shapes
:*
dtype0
|
Adam/h3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*!
shared_nameAdam/h3/kernel/v
u
$Adam/h3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/h3/kernel/v*
_output_shapes

:*
dtype0
t
Adam/h3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameAdam/h3/bias/v
m
"Adam/h3/bias/v/Read/ReadVariableOpReadVariableOpAdam/h3/bias/v*
_output_shapes
:*
dtype0
?
Adam/output/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*%
shared_nameAdam/output/kernel/v
}
(Adam/output/kernel/v/Read/ReadVariableOpReadVariableOpAdam/output/kernel/v*
_output_shapes

:*
dtype0
|
Adam/output/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/output/bias/v
u
&Adam/output/bias/v/Read/ReadVariableOpReadVariableOpAdam/output/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
?=
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?<
value?<B?< B?<
?
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer_with_weights-3
layer-7
	layer-8

layer_with_weights-4

layer-9
	optimizer
loss
trainable_variables
regularization_losses
	variables
	keras_api

signatures
 
h

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
R
trainable_variables
regularization_losses
	variables
	keras_api
h

kernel
bias
trainable_variables
regularization_losses
 	variables
!	keras_api
R
"trainable_variables
#regularization_losses
$	variables
%	keras_api
h

&kernel
'bias
(trainable_variables
)regularization_losses
*	variables
+	keras_api
R
,trainable_variables
-regularization_losses
.	variables
/	keras_api
h

0kernel
1bias
2trainable_variables
3regularization_losses
4	variables
5	keras_api
R
6trainable_variables
7regularization_losses
8	variables
9	keras_api
h

:kernel
;bias
<trainable_variables
=regularization_losses
>	variables
?	keras_api
?
@iter

Abeta_1

Bbeta_2
	Cdecay
Dlearning_ratem|m}m~m&m?'m?0m?1m?:m?;m?v?v?v?v?&v?'v?0v?1v?:v?;v?
 
F
0
1
2
3
&4
'5
06
17
:8
;9
 
F
0
1
2
3
&4
'5
06
17
:8
;9
?

Elayers
Fnon_trainable_variables
trainable_variables
regularization_losses
Glayer_metrics
Hlayer_regularization_losses
	variables
Imetrics
 
US
VARIABLE_VALUE	h0/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEh0/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
?

Jlayers
Knon_trainable_variables
trainable_variables
regularization_losses
Llayer_metrics
Mlayer_regularization_losses
	variables
Nmetrics
 
 
 
?

Olayers
Pnon_trainable_variables
trainable_variables
regularization_losses
Qlayer_metrics
Rlayer_regularization_losses
	variables
Smetrics
US
VARIABLE_VALUE	h1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEh1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
?

Tlayers
Unon_trainable_variables
trainable_variables
regularization_losses
Vlayer_metrics
Wlayer_regularization_losses
 	variables
Xmetrics
 
 
 
?

Ylayers
Znon_trainable_variables
"trainable_variables
#regularization_losses
[layer_metrics
\layer_regularization_losses
$	variables
]metrics
US
VARIABLE_VALUE	h2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEh2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

&0
'1
 

&0
'1
?

^layers
_non_trainable_variables
(trainable_variables
)regularization_losses
`layer_metrics
alayer_regularization_losses
*	variables
bmetrics
 
 
 
?

clayers
dnon_trainable_variables
,trainable_variables
-regularization_losses
elayer_metrics
flayer_regularization_losses
.	variables
gmetrics
US
VARIABLE_VALUE	h3/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEh3/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

00
11
 

00
11
?

hlayers
inon_trainable_variables
2trainable_variables
3regularization_losses
jlayer_metrics
klayer_regularization_losses
4	variables
lmetrics
 
 
 
?

mlayers
nnon_trainable_variables
6trainable_variables
7regularization_losses
olayer_metrics
player_regularization_losses
8	variables
qmetrics
YW
VARIABLE_VALUEoutput/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEoutput/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

:0
;1
 

:0
;1
?

rlayers
snon_trainable_variables
<trainable_variables
=regularization_losses
tlayer_metrics
ulayer_regularization_losses
>	variables
vmetrics
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
F
0
1
2
3
4
5
6
7
	8

9
 
 
 

w0
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
4
	xtotal
	ycount
z	variables
{	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

x0
y1

z	variables
xv
VARIABLE_VALUEAdam/h0/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEAdam/h0/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/h1/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEAdam/h1/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/h2/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEAdam/h2/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/h3/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEAdam/h3/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/output/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/output/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/h0/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEAdam/h0/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/h1/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEAdam/h1/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/h2/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEAdam/h2/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/h3/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEAdam/h3/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/output/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/output/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
x
serving_default_inputPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input	h0/kernelh0/bias	h1/kernelh1/bias	h2/kernelh2/bias	h3/kernelh3/biasoutput/kerneloutput/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *-
f(R&
$__inference_signature_wrapper_146445
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameh0/kernel/Read/ReadVariableOph0/bias/Read/ReadVariableOph1/kernel/Read/ReadVariableOph1/bias/Read/ReadVariableOph2/kernel/Read/ReadVariableOph2/bias/Read/ReadVariableOph3/kernel/Read/ReadVariableOph3/bias/Read/ReadVariableOp!output/kernel/Read/ReadVariableOpoutput/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp$Adam/h0/kernel/m/Read/ReadVariableOp"Adam/h0/bias/m/Read/ReadVariableOp$Adam/h1/kernel/m/Read/ReadVariableOp"Adam/h1/bias/m/Read/ReadVariableOp$Adam/h2/kernel/m/Read/ReadVariableOp"Adam/h2/bias/m/Read/ReadVariableOp$Adam/h3/kernel/m/Read/ReadVariableOp"Adam/h3/bias/m/Read/ReadVariableOp(Adam/output/kernel/m/Read/ReadVariableOp&Adam/output/bias/m/Read/ReadVariableOp$Adam/h0/kernel/v/Read/ReadVariableOp"Adam/h0/bias/v/Read/ReadVariableOp$Adam/h1/kernel/v/Read/ReadVariableOp"Adam/h1/bias/v/Read/ReadVariableOp$Adam/h2/kernel/v/Read/ReadVariableOp"Adam/h2/bias/v/Read/ReadVariableOp$Adam/h3/kernel/v/Read/ReadVariableOp"Adam/h3/bias/v/Read/ReadVariableOp(Adam/output/kernel/v/Read/ReadVariableOp&Adam/output/bias/v/Read/ReadVariableOpConst*2
Tin+
)2'	*
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
GPU 2J 8? *(
f#R!
__inference__traced_save_146944
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename	h0/kernelh0/bias	h1/kernelh1/bias	h2/kernelh2/bias	h3/kernelh3/biasoutput/kerneloutput/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcountAdam/h0/kernel/mAdam/h0/bias/mAdam/h1/kernel/mAdam/h1/bias/mAdam/h2/kernel/mAdam/h2/bias/mAdam/h3/kernel/mAdam/h3/bias/mAdam/output/kernel/mAdam/output/bias/mAdam/h0/kernel/vAdam/h0/bias/vAdam/h1/kernel/vAdam/h1/bias/vAdam/h2/kernel/vAdam/h2/bias/vAdam/h3/kernel/vAdam/h3/bias/vAdam/output/kernel/vAdam/output/bias/v*1
Tin*
(2&*
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
GPU 2J 8? *+
f&R$
"__inference__traced_restore_147065??
?
_
C__inference_dropout_layer_call_and_return_conditional_losses_146649

inputs
identityZ
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
"__inference__traced_restore_147065
file_prefix,
assignvariableop_h0_kernel:(
assignvariableop_1_h0_bias:.
assignvariableop_2_h1_kernel:(
assignvariableop_3_h1_bias:.
assignvariableop_4_h2_kernel:(
assignvariableop_5_h2_bias:.
assignvariableop_6_h3_kernel:(
assignvariableop_7_h3_bias:2
 assignvariableop_8_output_kernel:,
assignvariableop_9_output_bias:'
assignvariableop_10_adam_iter:	 )
assignvariableop_11_adam_beta_1: )
assignvariableop_12_adam_beta_2: (
assignvariableop_13_adam_decay: 0
&assignvariableop_14_adam_learning_rate: #
assignvariableop_15_total: #
assignvariableop_16_count: 6
$assignvariableop_17_adam_h0_kernel_m:0
"assignvariableop_18_adam_h0_bias_m:6
$assignvariableop_19_adam_h1_kernel_m:0
"assignvariableop_20_adam_h1_bias_m:6
$assignvariableop_21_adam_h2_kernel_m:0
"assignvariableop_22_adam_h2_bias_m:6
$assignvariableop_23_adam_h3_kernel_m:0
"assignvariableop_24_adam_h3_bias_m::
(assignvariableop_25_adam_output_kernel_m:4
&assignvariableop_26_adam_output_bias_m:6
$assignvariableop_27_adam_h0_kernel_v:0
"assignvariableop_28_adam_h0_bias_v:6
$assignvariableop_29_adam_h1_kernel_v:0
"assignvariableop_30_adam_h1_bias_v:6
$assignvariableop_31_adam_h2_kernel_v:0
"assignvariableop_32_adam_h2_bias_v:6
$assignvariableop_33_adam_h3_kernel_v:0
"assignvariableop_34_adam_h3_bias_v::
(assignvariableop_35_adam_output_kernel_v:4
&assignvariableop_36_adam_output_bias_v:
identity_38??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:&*
dtype0*?
value?B?&B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:&*
dtype0*_
valueVBT&B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::*4
dtypes*
(2&	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_h0_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_h0_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOpassignvariableop_2_h1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOpassignvariableop_3_h1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOpassignvariableop_4_h2_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOpassignvariableop_5_h2_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOpassignvariableop_6_h3_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOpassignvariableop_7_h3_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp assignvariableop_8_output_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOpassignvariableop_9_output_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_iterIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_beta_1Identity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOpassignvariableop_12_adam_beta_2Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOpassignvariableop_13_adam_decayIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp&assignvariableop_14_adam_learning_rateIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOpassignvariableop_15_totalIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOpassignvariableop_16_countIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp$assignvariableop_17_adam_h0_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp"assignvariableop_18_adam_h0_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp$assignvariableop_19_adam_h1_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp"assignvariableop_20_adam_h1_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp$assignvariableop_21_adam_h2_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp"assignvariableop_22_adam_h2_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp$assignvariableop_23_adam_h3_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp"assignvariableop_24_adam_h3_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp(assignvariableop_25_adam_output_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp&assignvariableop_26_adam_output_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp$assignvariableop_27_adam_h0_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp"assignvariableop_28_adam_h0_bias_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp$assignvariableop_29_adam_h1_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp"assignvariableop_30_adam_h1_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp$assignvariableop_31_adam_h2_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp"assignvariableop_32_adam_h2_bias_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp$assignvariableop_33_adam_h3_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp"assignvariableop_34_adam_h3_bias_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp(assignvariableop_35_adam_output_kernel_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp&assignvariableop_36_adam_output_bias_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_369
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_37Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_37?
Identity_38IdentityIdentity_37:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_38"#
identity_38Identity_38:output:0*_
_input_shapesN
L: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362(
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
?
?
>__inference_h1_layer_call_and_return_conditional_losses_146017

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:

identity_1??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoidb
mulMulBiasAdd:output:0Sigmoid:y:0*
T0*'
_output_shapes
:?????????2
mul[
IdentityIdentitymul:z:0*
T0*'
_output_shapes
:?????????2

Identity?
	IdentityN	IdentityNmul:z:0BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-146010*:
_output_shapes(
&:?????????:?????????2
	IdentityN?

Identity_1IdentityIdentityN:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
c
E__inference_dropout_3_layer_call_and_return_conditional_losses_146777

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
D
(__inference_dropout_layer_call_fn_146659

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_1462252
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?%
?
>__inference_nn_layer_call_and_return_conditional_losses_146298

inputs
	h0_146268:
	h0_146270:
	h1_146274:
	h1_146276:
	h2_146280:
	h2_146282:
	h3_146286:
	h3_146288:
output_146292:
output_146294:
identity??h0/StatefulPartitionedCall?h1/StatefulPartitionedCall?h2/StatefulPartitionedCall?h3/StatefulPartitionedCall?output/StatefulPartitionedCall?
h0/StatefulPartitionedCallStatefulPartitionedCallinputs	h0_146268	h0_146270*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *G
fBR@
>__inference_h0_layer_call_and_return_conditional_losses_1459882
h0/StatefulPartitionedCall?
dropout/PartitionedCallPartitionedCall#h0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_1462252
dropout/PartitionedCall?
h1/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0	h1_146274	h1_146276*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *G
fBR@
>__inference_h1_layer_call_and_return_conditional_losses_1460172
h1/StatefulPartitionedCall?
dropout_1/PartitionedCallPartitionedCall#h1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_1462002
dropout_1/PartitionedCall?
h2/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0	h2_146280	h2_146282*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *G
fBR@
>__inference_h2_layer_call_and_return_conditional_losses_1460462
h2/StatefulPartitionedCall?
dropout_2/PartitionedCallPartitionedCall#h2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_1461752
dropout_2/PartitionedCall?
h3/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0	h3_146286	h3_146288*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *G
fBR@
>__inference_h3_layer_call_and_return_conditional_losses_1460752
h3/StatefulPartitionedCall?
dropout_3/PartitionedCallPartitionedCall#h3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_3_layer_call_and_return_conditional_losses_1461502
dropout_3/PartitionedCall?
output/StatefulPartitionedCallStatefulPartitionedCall"dropout_3/PartitionedCall:output:0output_146292output_146294*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_1460982 
output/StatefulPartitionedCall?
IdentityIdentity'output/StatefulPartitionedCall:output:0^h0/StatefulPartitionedCall^h1/StatefulPartitionedCall^h2/StatefulPartitionedCall^h3/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : : : 28
h0/StatefulPartitionedCallh0/StatefulPartitionedCall28
h1/StatefulPartitionedCallh1/StatefulPartitionedCall28
h2/StatefulPartitionedCallh2/StatefulPartitionedCall28
h3/StatefulPartitionedCallh3/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
c
E__inference_dropout_3_layer_call_and_return_conditional_losses_146086

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
a
C__inference_dropout_layer_call_and_return_conditional_losses_146645

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
>__inference_h1_layer_call_and_return_conditional_losses_146675

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:

identity_1??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoidb
mulMulBiasAdd:output:0Sigmoid:y:0*
T0*'
_output_shapes
:?????????2
mul[
IdentityIdentitymul:z:0*
T0*'
_output_shapes
:?????????2

Identity?
	IdentityN	IdentityNmul:z:0BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-146668*:
_output_shapes(
&:?????????:?????????2
	IdentityN?

Identity_1IdentityIdentityN:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
a
E__inference_dropout_1_layer_call_and_return_conditional_losses_146693

inputs
identityZ
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
c
E__inference_dropout_1_layer_call_and_return_conditional_losses_146028

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
a
E__inference_dropout_2_layer_call_and_return_conditional_losses_146175

inputs
identityZ
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
#__inference_h1_layer_call_fn_146684

inputs
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *G
fBR@
>__inference_h1_layer_call_and_return_conditional_losses_1460172
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
#__inference_nn_layer_call_fn_146590

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *G
fBR@
>__inference_nn_layer_call_and_return_conditional_losses_1461052
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
D
(__inference_dropout_layer_call_fn_146654

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_1459992
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
F
*__inference_dropout_3_layer_call_fn_146791

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_3_layer_call_and_return_conditional_losses_1461502
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
F
*__inference_dropout_2_layer_call_fn_146747

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_1461752
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
B__inference_output_layer_call_and_return_conditional_losses_146801

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
c
E__inference_dropout_2_layer_call_and_return_conditional_losses_146057

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
a
C__inference_dropout_layer_call_and_return_conditional_losses_145999

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
_
C__inference_dropout_layer_call_and_return_conditional_losses_146225

inputs
identityZ
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
#__inference_h0_layer_call_fn_146640

inputs
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *G
fBR@
>__inference_h0_layer_call_and_return_conditional_losses_1459882
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
a
E__inference_dropout_1_layer_call_and_return_conditional_losses_146200

inputs
identityZ
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
>__inference_h0_layer_call_and_return_conditional_losses_146631

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:

identity_1??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoidb
mulMulBiasAdd:output:0Sigmoid:y:0*
T0*'
_output_shapes
:?????????2
mul[
IdentityIdentitymul:z:0*
T0*'
_output_shapes
:?????????2

Identity?
	IdentityN	IdentityNmul:z:0BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-146624*:
_output_shapes(
&:?????????:?????????2
	IdentityN?

Identity_1IdentityIdentityN:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
>__inference_h2_layer_call_and_return_conditional_losses_146046

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:

identity_1??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoidb
mulMulBiasAdd:output:0Sigmoid:y:0*
T0*'
_output_shapes
:?????????2
mul[
IdentityIdentitymul:z:0*
T0*'
_output_shapes
:?????????2

Identity?
	IdentityN	IdentityNmul:z:0BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-146039*:
_output_shapes(
&:?????????:?????????2
	IdentityN?

Identity_1IdentityIdentityN:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?L
?
__inference__traced_save_146944
file_prefix(
$savev2_h0_kernel_read_readvariableop&
"savev2_h0_bias_read_readvariableop(
$savev2_h1_kernel_read_readvariableop&
"savev2_h1_bias_read_readvariableop(
$savev2_h2_kernel_read_readvariableop&
"savev2_h2_bias_read_readvariableop(
$savev2_h3_kernel_read_readvariableop&
"savev2_h3_bias_read_readvariableop,
(savev2_output_kernel_read_readvariableop*
&savev2_output_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop/
+savev2_adam_h0_kernel_m_read_readvariableop-
)savev2_adam_h0_bias_m_read_readvariableop/
+savev2_adam_h1_kernel_m_read_readvariableop-
)savev2_adam_h1_bias_m_read_readvariableop/
+savev2_adam_h2_kernel_m_read_readvariableop-
)savev2_adam_h2_bias_m_read_readvariableop/
+savev2_adam_h3_kernel_m_read_readvariableop-
)savev2_adam_h3_bias_m_read_readvariableop3
/savev2_adam_output_kernel_m_read_readvariableop1
-savev2_adam_output_bias_m_read_readvariableop/
+savev2_adam_h0_kernel_v_read_readvariableop-
)savev2_adam_h0_bias_v_read_readvariableop/
+savev2_adam_h1_kernel_v_read_readvariableop-
)savev2_adam_h1_bias_v_read_readvariableop/
+savev2_adam_h2_kernel_v_read_readvariableop-
)savev2_adam_h2_bias_v_read_readvariableop/
+savev2_adam_h3_kernel_v_read_readvariableop-
)savev2_adam_h3_bias_v_read_readvariableop3
/savev2_adam_output_kernel_v_read_readvariableop1
-savev2_adam_output_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
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
Const_1?
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
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:&*
dtype0*?
value?B?&B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:&*
dtype0*_
valueVBT&B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0$savev2_h0_kernel_read_readvariableop"savev2_h0_bias_read_readvariableop$savev2_h1_kernel_read_readvariableop"savev2_h1_bias_read_readvariableop$savev2_h2_kernel_read_readvariableop"savev2_h2_bias_read_readvariableop$savev2_h3_kernel_read_readvariableop"savev2_h3_bias_read_readvariableop(savev2_output_kernel_read_readvariableop&savev2_output_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop+savev2_adam_h0_kernel_m_read_readvariableop)savev2_adam_h0_bias_m_read_readvariableop+savev2_adam_h1_kernel_m_read_readvariableop)savev2_adam_h1_bias_m_read_readvariableop+savev2_adam_h2_kernel_m_read_readvariableop)savev2_adam_h2_bias_m_read_readvariableop+savev2_adam_h3_kernel_m_read_readvariableop)savev2_adam_h3_bias_m_read_readvariableop/savev2_adam_output_kernel_m_read_readvariableop-savev2_adam_output_bias_m_read_readvariableop+savev2_adam_h0_kernel_v_read_readvariableop)savev2_adam_h0_bias_v_read_readvariableop+savev2_adam_h1_kernel_v_read_readvariableop)savev2_adam_h1_bias_v_read_readvariableop+savev2_adam_h2_kernel_v_read_readvariableop)savev2_adam_h2_bias_v_read_readvariableop+savev2_adam_h3_kernel_v_read_readvariableop)savev2_adam_h3_bias_v_read_readvariableop/savev2_adam_output_kernel_v_read_readvariableop-savev2_adam_output_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *4
dtypes*
(2&	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
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

identity_1Identity_1:output:0*?
_input_shapes?
?: ::::::::::: : : : : : : ::::::::::::::::::::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$	 

_output_shapes

:: 


_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$  

_output_shapes

:: !

_output_shapes
::$" 

_output_shapes

:: #

_output_shapes
::$$ 

_output_shapes

:: %

_output_shapes
::&

_output_shapes
: 
?
?
#__inference_h3_layer_call_fn_146772

inputs
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *G
fBR@
>__inference_h3_layer_call_and_return_conditional_losses_1460752
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
'__inference_output_layer_call_fn_146810

inputs
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_1460982
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
a
E__inference_dropout_3_layer_call_and_return_conditional_losses_146150

inputs
identityZ
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
c
E__inference_dropout_1_layer_call_and_return_conditional_losses_146689

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
F
*__inference_dropout_3_layer_call_fn_146786

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_3_layer_call_and_return_conditional_losses_1460862
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?%
?
>__inference_nn_layer_call_and_return_conditional_losses_146379	
input
	h0_146349:
	h0_146351:
	h1_146355:
	h1_146357:
	h2_146361:
	h2_146363:
	h3_146367:
	h3_146369:
output_146373:
output_146375:
identity??h0/StatefulPartitionedCall?h1/StatefulPartitionedCall?h2/StatefulPartitionedCall?h3/StatefulPartitionedCall?output/StatefulPartitionedCall?
h0/StatefulPartitionedCallStatefulPartitionedCallinput	h0_146349	h0_146351*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *G
fBR@
>__inference_h0_layer_call_and_return_conditional_losses_1459882
h0/StatefulPartitionedCall?
dropout/PartitionedCallPartitionedCall#h0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_1459992
dropout/PartitionedCall?
h1/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0	h1_146355	h1_146357*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *G
fBR@
>__inference_h1_layer_call_and_return_conditional_losses_1460172
h1/StatefulPartitionedCall?
dropout_1/PartitionedCallPartitionedCall#h1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_1460282
dropout_1/PartitionedCall?
h2/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0	h2_146361	h2_146363*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *G
fBR@
>__inference_h2_layer_call_and_return_conditional_losses_1460462
h2/StatefulPartitionedCall?
dropout_2/PartitionedCallPartitionedCall#h2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_1460572
dropout_2/PartitionedCall?
h3/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0	h3_146367	h3_146369*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *G
fBR@
>__inference_h3_layer_call_and_return_conditional_losses_1460752
h3/StatefulPartitionedCall?
dropout_3/PartitionedCallPartitionedCall#h3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_3_layer_call_and_return_conditional_losses_1460862
dropout_3/PartitionedCall?
output/StatefulPartitionedCallStatefulPartitionedCall"dropout_3/PartitionedCall:output:0output_146373output_146375*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_1460982 
output/StatefulPartitionedCall?
IdentityIdentity'output/StatefulPartitionedCall:output:0^h0/StatefulPartitionedCall^h1/StatefulPartitionedCall^h2/StatefulPartitionedCall^h3/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : : : 28
h0/StatefulPartitionedCallh0/StatefulPartitionedCall28
h1/StatefulPartitionedCallh1/StatefulPartitionedCall28
h2/StatefulPartitionedCallh2/StatefulPartitionedCall28
h3/StatefulPartitionedCallh3/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:N J
'
_output_shapes
:?????????

_user_specified_nameinput
?
F
*__inference_dropout_1_layer_call_fn_146698

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_1460282
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
>__inference_h2_layer_call_and_return_conditional_losses_146719

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:

identity_1??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoidb
mulMulBiasAdd:output:0Sigmoid:y:0*
T0*'
_output_shapes
:?????????2
mul[
IdentityIdentitymul:z:0*
T0*'
_output_shapes
:?????????2

Identity?
	IdentityN	IdentityNmul:z:0BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-146712*:
_output_shapes(
&:?????????:?????????2
	IdentityN?

Identity_1IdentityIdentityN:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
B__inference_output_layer_call_and_return_conditional_losses_146098

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
F
*__inference_dropout_2_layer_call_fn_146742

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_1460572
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?@
?
!__inference__wrapped_model_145965	
input6
$nn_h0_matmul_readvariableop_resource:3
%nn_h0_biasadd_readvariableop_resource:6
$nn_h1_matmul_readvariableop_resource:3
%nn_h1_biasadd_readvariableop_resource:6
$nn_h2_matmul_readvariableop_resource:3
%nn_h2_biasadd_readvariableop_resource:6
$nn_h3_matmul_readvariableop_resource:3
%nn_h3_biasadd_readvariableop_resource::
(nn_output_matmul_readvariableop_resource:7
)nn_output_biasadd_readvariableop_resource:
identity??nn/h0/BiasAdd/ReadVariableOp?nn/h0/MatMul/ReadVariableOp?nn/h1/BiasAdd/ReadVariableOp?nn/h1/MatMul/ReadVariableOp?nn/h2/BiasAdd/ReadVariableOp?nn/h2/MatMul/ReadVariableOp?nn/h3/BiasAdd/ReadVariableOp?nn/h3/MatMul/ReadVariableOp? nn/output/BiasAdd/ReadVariableOp?nn/output/MatMul/ReadVariableOp?
nn/h0/MatMul/ReadVariableOpReadVariableOp$nn_h0_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
nn/h0/MatMul/ReadVariableOp?
nn/h0/MatMulMatMulinput#nn/h0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
nn/h0/MatMul?
nn/h0/BiasAdd/ReadVariableOpReadVariableOp%nn_h0_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
nn/h0/BiasAdd/ReadVariableOp?
nn/h0/BiasAddBiasAddnn/h0/MatMul:product:0$nn/h0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
nn/h0/BiasAdds
nn/h0/SigmoidSigmoidnn/h0/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
nn/h0/Sigmoidz
	nn/h0/mulMulnn/h0/BiasAdd:output:0nn/h0/Sigmoid:y:0*
T0*'
_output_shapes
:?????????2
	nn/h0/mulm
nn/h0/IdentityIdentitynn/h0/mul:z:0*
T0*'
_output_shapes
:?????????2
nn/h0/Identity?
nn/h0/IdentityN	IdentityNnn/h0/mul:z:0nn/h0/BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-145912*:
_output_shapes(
&:?????????:?????????2
nn/h0/IdentityN?
nn/dropout/IdentityIdentitynn/h0/IdentityN:output:0*
T0*'
_output_shapes
:?????????2
nn/dropout/Identity?
nn/h1/MatMul/ReadVariableOpReadVariableOp$nn_h1_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
nn/h1/MatMul/ReadVariableOp?
nn/h1/MatMulMatMulnn/dropout/Identity:output:0#nn/h1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
nn/h1/MatMul?
nn/h1/BiasAdd/ReadVariableOpReadVariableOp%nn_h1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
nn/h1/BiasAdd/ReadVariableOp?
nn/h1/BiasAddBiasAddnn/h1/MatMul:product:0$nn/h1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
nn/h1/BiasAdds
nn/h1/SigmoidSigmoidnn/h1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
nn/h1/Sigmoidz
	nn/h1/mulMulnn/h1/BiasAdd:output:0nn/h1/Sigmoid:y:0*
T0*'
_output_shapes
:?????????2
	nn/h1/mulm
nn/h1/IdentityIdentitynn/h1/mul:z:0*
T0*'
_output_shapes
:?????????2
nn/h1/Identity?
nn/h1/IdentityN	IdentityNnn/h1/mul:z:0nn/h1/BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-145925*:
_output_shapes(
&:?????????:?????????2
nn/h1/IdentityN?
nn/dropout_1/IdentityIdentitynn/h1/IdentityN:output:0*
T0*'
_output_shapes
:?????????2
nn/dropout_1/Identity?
nn/h2/MatMul/ReadVariableOpReadVariableOp$nn_h2_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
nn/h2/MatMul/ReadVariableOp?
nn/h2/MatMulMatMulnn/dropout_1/Identity:output:0#nn/h2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
nn/h2/MatMul?
nn/h2/BiasAdd/ReadVariableOpReadVariableOp%nn_h2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
nn/h2/BiasAdd/ReadVariableOp?
nn/h2/BiasAddBiasAddnn/h2/MatMul:product:0$nn/h2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
nn/h2/BiasAdds
nn/h2/SigmoidSigmoidnn/h2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
nn/h2/Sigmoidz
	nn/h2/mulMulnn/h2/BiasAdd:output:0nn/h2/Sigmoid:y:0*
T0*'
_output_shapes
:?????????2
	nn/h2/mulm
nn/h2/IdentityIdentitynn/h2/mul:z:0*
T0*'
_output_shapes
:?????????2
nn/h2/Identity?
nn/h2/IdentityN	IdentityNnn/h2/mul:z:0nn/h2/BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-145938*:
_output_shapes(
&:?????????:?????????2
nn/h2/IdentityN?
nn/dropout_2/IdentityIdentitynn/h2/IdentityN:output:0*
T0*'
_output_shapes
:?????????2
nn/dropout_2/Identity?
nn/h3/MatMul/ReadVariableOpReadVariableOp$nn_h3_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
nn/h3/MatMul/ReadVariableOp?
nn/h3/MatMulMatMulnn/dropout_2/Identity:output:0#nn/h3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
nn/h3/MatMul?
nn/h3/BiasAdd/ReadVariableOpReadVariableOp%nn_h3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
nn/h3/BiasAdd/ReadVariableOp?
nn/h3/BiasAddBiasAddnn/h3/MatMul:product:0$nn/h3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
nn/h3/BiasAdds
nn/h3/SigmoidSigmoidnn/h3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
nn/h3/Sigmoidz
	nn/h3/mulMulnn/h3/BiasAdd:output:0nn/h3/Sigmoid:y:0*
T0*'
_output_shapes
:?????????2
	nn/h3/mulm
nn/h3/IdentityIdentitynn/h3/mul:z:0*
T0*'
_output_shapes
:?????????2
nn/h3/Identity?
nn/h3/IdentityN	IdentityNnn/h3/mul:z:0nn/h3/BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-145951*:
_output_shapes(
&:?????????:?????????2
nn/h3/IdentityN?
nn/dropout_3/IdentityIdentitynn/h3/IdentityN:output:0*
T0*'
_output_shapes
:?????????2
nn/dropout_3/Identity?
nn/output/MatMul/ReadVariableOpReadVariableOp(nn_output_matmul_readvariableop_resource*
_output_shapes

:*
dtype02!
nn/output/MatMul/ReadVariableOp?
nn/output/MatMulMatMulnn/dropout_3/Identity:output:0'nn/output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
nn/output/MatMul?
 nn/output/BiasAdd/ReadVariableOpReadVariableOp)nn_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 nn/output/BiasAdd/ReadVariableOp?
nn/output/BiasAddBiasAddnn/output/MatMul:product:0(nn/output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
nn/output/BiasAdd?
IdentityIdentitynn/output/BiasAdd:output:0^nn/h0/BiasAdd/ReadVariableOp^nn/h0/MatMul/ReadVariableOp^nn/h1/BiasAdd/ReadVariableOp^nn/h1/MatMul/ReadVariableOp^nn/h2/BiasAdd/ReadVariableOp^nn/h2/MatMul/ReadVariableOp^nn/h3/BiasAdd/ReadVariableOp^nn/h3/MatMul/ReadVariableOp!^nn/output/BiasAdd/ReadVariableOp ^nn/output/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : : : 2<
nn/h0/BiasAdd/ReadVariableOpnn/h0/BiasAdd/ReadVariableOp2:
nn/h0/MatMul/ReadVariableOpnn/h0/MatMul/ReadVariableOp2<
nn/h1/BiasAdd/ReadVariableOpnn/h1/BiasAdd/ReadVariableOp2:
nn/h1/MatMul/ReadVariableOpnn/h1/MatMul/ReadVariableOp2<
nn/h2/BiasAdd/ReadVariableOpnn/h2/BiasAdd/ReadVariableOp2:
nn/h2/MatMul/ReadVariableOpnn/h2/MatMul/ReadVariableOp2<
nn/h3/BiasAdd/ReadVariableOpnn/h3/BiasAdd/ReadVariableOp2:
nn/h3/MatMul/ReadVariableOpnn/h3/MatMul/ReadVariableOp2D
 nn/output/BiasAdd/ReadVariableOp nn/output/BiasAdd/ReadVariableOp2B
nn/output/MatMul/ReadVariableOpnn/output/MatMul/ReadVariableOp:N J
'
_output_shapes
:?????????

_user_specified_nameinput
?
a
E__inference_dropout_3_layer_call_and_return_conditional_losses_146781

inputs
identityZ
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
$__inference_signature_wrapper_146445	
input
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__wrapped_model_1459652
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:N J
'
_output_shapes
:?????????

_user_specified_nameinput
?
a
E__inference_dropout_2_layer_call_and_return_conditional_losses_146737

inputs
identityZ
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?7
?
>__inference_nn_layer_call_and_return_conditional_losses_146565

inputs3
!h0_matmul_readvariableop_resource:0
"h0_biasadd_readvariableop_resource:3
!h1_matmul_readvariableop_resource:0
"h1_biasadd_readvariableop_resource:3
!h2_matmul_readvariableop_resource:0
"h2_biasadd_readvariableop_resource:3
!h3_matmul_readvariableop_resource:0
"h3_biasadd_readvariableop_resource:7
%output_matmul_readvariableop_resource:4
&output_biasadd_readvariableop_resource:
identity??h0/BiasAdd/ReadVariableOp?h0/MatMul/ReadVariableOp?h1/BiasAdd/ReadVariableOp?h1/MatMul/ReadVariableOp?h2/BiasAdd/ReadVariableOp?h2/MatMul/ReadVariableOp?h3/BiasAdd/ReadVariableOp?h3/MatMul/ReadVariableOp?output/BiasAdd/ReadVariableOp?output/MatMul/ReadVariableOp?
h0/MatMul/ReadVariableOpReadVariableOp!h0_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
h0/MatMul/ReadVariableOp|
	h0/MatMulMatMulinputs h0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
	h0/MatMul?
h0/BiasAdd/ReadVariableOpReadVariableOp"h0_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
h0/BiasAdd/ReadVariableOp?

h0/BiasAddBiasAddh0/MatMul:product:0!h0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2

h0/BiasAddj

h0/SigmoidSigmoidh0/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2

h0/Sigmoidn
h0/mulMulh0/BiasAdd:output:0h0/Sigmoid:y:0*
T0*'
_output_shapes
:?????????2
h0/muld
h0/IdentityIdentity
h0/mul:z:0*
T0*'
_output_shapes
:?????????2
h0/Identity?
h0/IdentityN	IdentityN
h0/mul:z:0h0/BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-146516*:
_output_shapes(
&:?????????:?????????2
h0/IdentityN?
h1/MatMul/ReadVariableOpReadVariableOp!h1_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
h1/MatMul/ReadVariableOp?
	h1/MatMulMatMulh0/IdentityN:output:0 h1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
	h1/MatMul?
h1/BiasAdd/ReadVariableOpReadVariableOp"h1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
h1/BiasAdd/ReadVariableOp?

h1/BiasAddBiasAddh1/MatMul:product:0!h1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2

h1/BiasAddj

h1/SigmoidSigmoidh1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2

h1/Sigmoidn
h1/mulMulh1/BiasAdd:output:0h1/Sigmoid:y:0*
T0*'
_output_shapes
:?????????2
h1/muld
h1/IdentityIdentity
h1/mul:z:0*
T0*'
_output_shapes
:?????????2
h1/Identity?
h1/IdentityN	IdentityN
h1/mul:z:0h1/BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-146528*:
_output_shapes(
&:?????????:?????????2
h1/IdentityN?
h2/MatMul/ReadVariableOpReadVariableOp!h2_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
h2/MatMul/ReadVariableOp?
	h2/MatMulMatMulh1/IdentityN:output:0 h2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
	h2/MatMul?
h2/BiasAdd/ReadVariableOpReadVariableOp"h2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
h2/BiasAdd/ReadVariableOp?

h2/BiasAddBiasAddh2/MatMul:product:0!h2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2

h2/BiasAddj

h2/SigmoidSigmoidh2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2

h2/Sigmoidn
h2/mulMulh2/BiasAdd:output:0h2/Sigmoid:y:0*
T0*'
_output_shapes
:?????????2
h2/muld
h2/IdentityIdentity
h2/mul:z:0*
T0*'
_output_shapes
:?????????2
h2/Identity?
h2/IdentityN	IdentityN
h2/mul:z:0h2/BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-146540*:
_output_shapes(
&:?????????:?????????2
h2/IdentityN?
h3/MatMul/ReadVariableOpReadVariableOp!h3_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
h3/MatMul/ReadVariableOp?
	h3/MatMulMatMulh2/IdentityN:output:0 h3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
	h3/MatMul?
h3/BiasAdd/ReadVariableOpReadVariableOp"h3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
h3/BiasAdd/ReadVariableOp?

h3/BiasAddBiasAddh3/MatMul:product:0!h3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2

h3/BiasAddj

h3/SigmoidSigmoidh3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2

h3/Sigmoidn
h3/mulMulh3/BiasAdd:output:0h3/Sigmoid:y:0*
T0*'
_output_shapes
:?????????2
h3/muld
h3/IdentityIdentity
h3/mul:z:0*
T0*'
_output_shapes
:?????????2
h3/Identity?
h3/IdentityN	IdentityN
h3/mul:z:0h3/BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-146552*:
_output_shapes(
&:?????????:?????????2
h3/IdentityN?
output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
output/MatMul/ReadVariableOp?
output/MatMulMatMulh3/IdentityN:output:0$output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
output/MatMul?
output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
output/BiasAdd/ReadVariableOp?
output/BiasAddBiasAddoutput/MatMul:product:0%output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
output/BiasAdd?
IdentityIdentityoutput/BiasAdd:output:0^h0/BiasAdd/ReadVariableOp^h0/MatMul/ReadVariableOp^h1/BiasAdd/ReadVariableOp^h1/MatMul/ReadVariableOp^h2/BiasAdd/ReadVariableOp^h2/MatMul/ReadVariableOp^h3/BiasAdd/ReadVariableOp^h3/MatMul/ReadVariableOp^output/BiasAdd/ReadVariableOp^output/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : : : 26
h0/BiasAdd/ReadVariableOph0/BiasAdd/ReadVariableOp24
h0/MatMul/ReadVariableOph0/MatMul/ReadVariableOp26
h1/BiasAdd/ReadVariableOph1/BiasAdd/ReadVariableOp24
h1/MatMul/ReadVariableOph1/MatMul/ReadVariableOp26
h2/BiasAdd/ReadVariableOph2/BiasAdd/ReadVariableOp24
h2/MatMul/ReadVariableOph2/MatMul/ReadVariableOp26
h3/BiasAdd/ReadVariableOph3/BiasAdd/ReadVariableOp24
h3/MatMul/ReadVariableOph3/MatMul/ReadVariableOp2>
output/BiasAdd/ReadVariableOpoutput/BiasAdd/ReadVariableOp2<
output/MatMul/ReadVariableOpoutput/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?%
?
>__inference_nn_layer_call_and_return_conditional_losses_146412	
input
	h0_146382:
	h0_146384:
	h1_146388:
	h1_146390:
	h2_146394:
	h2_146396:
	h3_146400:
	h3_146402:
output_146406:
output_146408:
identity??h0/StatefulPartitionedCall?h1/StatefulPartitionedCall?h2/StatefulPartitionedCall?h3/StatefulPartitionedCall?output/StatefulPartitionedCall?
h0/StatefulPartitionedCallStatefulPartitionedCallinput	h0_146382	h0_146384*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *G
fBR@
>__inference_h0_layer_call_and_return_conditional_losses_1459882
h0/StatefulPartitionedCall?
dropout/PartitionedCallPartitionedCall#h0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_1462252
dropout/PartitionedCall?
h1/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0	h1_146388	h1_146390*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *G
fBR@
>__inference_h1_layer_call_and_return_conditional_losses_1460172
h1/StatefulPartitionedCall?
dropout_1/PartitionedCallPartitionedCall#h1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_1462002
dropout_1/PartitionedCall?
h2/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0	h2_146394	h2_146396*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *G
fBR@
>__inference_h2_layer_call_and_return_conditional_losses_1460462
h2/StatefulPartitionedCall?
dropout_2/PartitionedCallPartitionedCall#h2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_1461752
dropout_2/PartitionedCall?
h3/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0	h3_146400	h3_146402*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *G
fBR@
>__inference_h3_layer_call_and_return_conditional_losses_1460752
h3/StatefulPartitionedCall?
dropout_3/PartitionedCallPartitionedCall#h3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_3_layer_call_and_return_conditional_losses_1461502
dropout_3/PartitionedCall?
output/StatefulPartitionedCallStatefulPartitionedCall"dropout_3/PartitionedCall:output:0output_146406output_146408*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_1460982 
output/StatefulPartitionedCall?
IdentityIdentity'output/StatefulPartitionedCall:output:0^h0/StatefulPartitionedCall^h1/StatefulPartitionedCall^h2/StatefulPartitionedCall^h3/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : : : 28
h0/StatefulPartitionedCallh0/StatefulPartitionedCall28
h1/StatefulPartitionedCallh1/StatefulPartitionedCall28
h2/StatefulPartitionedCallh2/StatefulPartitionedCall28
h3/StatefulPartitionedCallh3/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:N J
'
_output_shapes
:?????????

_user_specified_nameinput
?	
?
#__inference_nn_layer_call_fn_146615

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *G
fBR@
>__inference_nn_layer_call_and_return_conditional_losses_1462982
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
>__inference_h3_layer_call_and_return_conditional_losses_146075

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:

identity_1??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoidb
mulMulBiasAdd:output:0Sigmoid:y:0*
T0*'
_output_shapes
:?????????2
mul[
IdentityIdentitymul:z:0*
T0*'
_output_shapes
:?????????2

Identity?
	IdentityN	IdentityNmul:z:0BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-146068*:
_output_shapes(
&:?????????:?????????2
	IdentityN?

Identity_1IdentityIdentityN:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
#__inference_h2_layer_call_fn_146728

inputs
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *G
fBR@
>__inference_h2_layer_call_and_return_conditional_losses_1460462
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
c
E__inference_dropout_2_layer_call_and_return_conditional_losses_146733

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
#__inference_nn_layer_call_fn_146128	
input
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *G
fBR@
>__inference_nn_layer_call_and_return_conditional_losses_1461052
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:N J
'
_output_shapes
:?????????

_user_specified_nameinput
?%
?
>__inference_nn_layer_call_and_return_conditional_losses_146105

inputs
	h0_145989:
	h0_145991:
	h1_146018:
	h1_146020:
	h2_146047:
	h2_146049:
	h3_146076:
	h3_146078:
output_146099:
output_146101:
identity??h0/StatefulPartitionedCall?h1/StatefulPartitionedCall?h2/StatefulPartitionedCall?h3/StatefulPartitionedCall?output/StatefulPartitionedCall?
h0/StatefulPartitionedCallStatefulPartitionedCallinputs	h0_145989	h0_145991*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *G
fBR@
>__inference_h0_layer_call_and_return_conditional_losses_1459882
h0/StatefulPartitionedCall?
dropout/PartitionedCallPartitionedCall#h0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_1459992
dropout/PartitionedCall?
h1/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0	h1_146018	h1_146020*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *G
fBR@
>__inference_h1_layer_call_and_return_conditional_losses_1460172
h1/StatefulPartitionedCall?
dropout_1/PartitionedCallPartitionedCall#h1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_1460282
dropout_1/PartitionedCall?
h2/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0	h2_146047	h2_146049*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *G
fBR@
>__inference_h2_layer_call_and_return_conditional_losses_1460462
h2/StatefulPartitionedCall?
dropout_2/PartitionedCallPartitionedCall#h2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_1460572
dropout_2/PartitionedCall?
h3/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0	h3_146076	h3_146078*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *G
fBR@
>__inference_h3_layer_call_and_return_conditional_losses_1460752
h3/StatefulPartitionedCall?
dropout_3/PartitionedCallPartitionedCall#h3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_3_layer_call_and_return_conditional_losses_1460862
dropout_3/PartitionedCall?
output/StatefulPartitionedCallStatefulPartitionedCall"dropout_3/PartitionedCall:output:0output_146099output_146101*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_1460982 
output/StatefulPartitionedCall?
IdentityIdentity'output/StatefulPartitionedCall:output:0^h0/StatefulPartitionedCall^h1/StatefulPartitionedCall^h2/StatefulPartitionedCall^h3/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : : : 28
h0/StatefulPartitionedCallh0/StatefulPartitionedCall28
h1/StatefulPartitionedCallh1/StatefulPartitionedCall28
h2/StatefulPartitionedCallh2/StatefulPartitionedCall28
h3/StatefulPartitionedCallh3/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?;
?
>__inference_nn_layer_call_and_return_conditional_losses_146507

inputs3
!h0_matmul_readvariableop_resource:0
"h0_biasadd_readvariableop_resource:3
!h1_matmul_readvariableop_resource:0
"h1_biasadd_readvariableop_resource:3
!h2_matmul_readvariableop_resource:0
"h2_biasadd_readvariableop_resource:3
!h3_matmul_readvariableop_resource:0
"h3_biasadd_readvariableop_resource:7
%output_matmul_readvariableop_resource:4
&output_biasadd_readvariableop_resource:
identity??h0/BiasAdd/ReadVariableOp?h0/MatMul/ReadVariableOp?h1/BiasAdd/ReadVariableOp?h1/MatMul/ReadVariableOp?h2/BiasAdd/ReadVariableOp?h2/MatMul/ReadVariableOp?h3/BiasAdd/ReadVariableOp?h3/MatMul/ReadVariableOp?output/BiasAdd/ReadVariableOp?output/MatMul/ReadVariableOp?
h0/MatMul/ReadVariableOpReadVariableOp!h0_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
h0/MatMul/ReadVariableOp|
	h0/MatMulMatMulinputs h0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
	h0/MatMul?
h0/BiasAdd/ReadVariableOpReadVariableOp"h0_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
h0/BiasAdd/ReadVariableOp?

h0/BiasAddBiasAddh0/MatMul:product:0!h0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2

h0/BiasAddj

h0/SigmoidSigmoidh0/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2

h0/Sigmoidn
h0/mulMulh0/BiasAdd:output:0h0/Sigmoid:y:0*
T0*'
_output_shapes
:?????????2
h0/muld
h0/IdentityIdentity
h0/mul:z:0*
T0*'
_output_shapes
:?????????2
h0/Identity?
h0/IdentityN	IdentityN
h0/mul:z:0h0/BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-146454*:
_output_shapes(
&:?????????:?????????2
h0/IdentityNy
dropout/IdentityIdentityh0/IdentityN:output:0*
T0*'
_output_shapes
:?????????2
dropout/Identity?
h1/MatMul/ReadVariableOpReadVariableOp!h1_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
h1/MatMul/ReadVariableOp?
	h1/MatMulMatMuldropout/Identity:output:0 h1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
	h1/MatMul?
h1/BiasAdd/ReadVariableOpReadVariableOp"h1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
h1/BiasAdd/ReadVariableOp?

h1/BiasAddBiasAddh1/MatMul:product:0!h1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2

h1/BiasAddj

h1/SigmoidSigmoidh1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2

h1/Sigmoidn
h1/mulMulh1/BiasAdd:output:0h1/Sigmoid:y:0*
T0*'
_output_shapes
:?????????2
h1/muld
h1/IdentityIdentity
h1/mul:z:0*
T0*'
_output_shapes
:?????????2
h1/Identity?
h1/IdentityN	IdentityN
h1/mul:z:0h1/BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-146467*:
_output_shapes(
&:?????????:?????????2
h1/IdentityN}
dropout_1/IdentityIdentityh1/IdentityN:output:0*
T0*'
_output_shapes
:?????????2
dropout_1/Identity?
h2/MatMul/ReadVariableOpReadVariableOp!h2_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
h2/MatMul/ReadVariableOp?
	h2/MatMulMatMuldropout_1/Identity:output:0 h2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
	h2/MatMul?
h2/BiasAdd/ReadVariableOpReadVariableOp"h2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
h2/BiasAdd/ReadVariableOp?

h2/BiasAddBiasAddh2/MatMul:product:0!h2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2

h2/BiasAddj

h2/SigmoidSigmoidh2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2

h2/Sigmoidn
h2/mulMulh2/BiasAdd:output:0h2/Sigmoid:y:0*
T0*'
_output_shapes
:?????????2
h2/muld
h2/IdentityIdentity
h2/mul:z:0*
T0*'
_output_shapes
:?????????2
h2/Identity?
h2/IdentityN	IdentityN
h2/mul:z:0h2/BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-146480*:
_output_shapes(
&:?????????:?????????2
h2/IdentityN}
dropout_2/IdentityIdentityh2/IdentityN:output:0*
T0*'
_output_shapes
:?????????2
dropout_2/Identity?
h3/MatMul/ReadVariableOpReadVariableOp!h3_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
h3/MatMul/ReadVariableOp?
	h3/MatMulMatMuldropout_2/Identity:output:0 h3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
	h3/MatMul?
h3/BiasAdd/ReadVariableOpReadVariableOp"h3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
h3/BiasAdd/ReadVariableOp?

h3/BiasAddBiasAddh3/MatMul:product:0!h3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2

h3/BiasAddj

h3/SigmoidSigmoidh3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2

h3/Sigmoidn
h3/mulMulh3/BiasAdd:output:0h3/Sigmoid:y:0*
T0*'
_output_shapes
:?????????2
h3/muld
h3/IdentityIdentity
h3/mul:z:0*
T0*'
_output_shapes
:?????????2
h3/Identity?
h3/IdentityN	IdentityN
h3/mul:z:0h3/BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-146493*:
_output_shapes(
&:?????????:?????????2
h3/IdentityN}
dropout_3/IdentityIdentityh3/IdentityN:output:0*
T0*'
_output_shapes
:?????????2
dropout_3/Identity?
output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
output/MatMul/ReadVariableOp?
output/MatMulMatMuldropout_3/Identity:output:0$output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
output/MatMul?
output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
output/BiasAdd/ReadVariableOp?
output/BiasAddBiasAddoutput/MatMul:product:0%output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
output/BiasAdd?
IdentityIdentityoutput/BiasAdd:output:0^h0/BiasAdd/ReadVariableOp^h0/MatMul/ReadVariableOp^h1/BiasAdd/ReadVariableOp^h1/MatMul/ReadVariableOp^h2/BiasAdd/ReadVariableOp^h2/MatMul/ReadVariableOp^h3/BiasAdd/ReadVariableOp^h3/MatMul/ReadVariableOp^output/BiasAdd/ReadVariableOp^output/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : : : 26
h0/BiasAdd/ReadVariableOph0/BiasAdd/ReadVariableOp24
h0/MatMul/ReadVariableOph0/MatMul/ReadVariableOp26
h1/BiasAdd/ReadVariableOph1/BiasAdd/ReadVariableOp24
h1/MatMul/ReadVariableOph1/MatMul/ReadVariableOp26
h2/BiasAdd/ReadVariableOph2/BiasAdd/ReadVariableOp24
h2/MatMul/ReadVariableOph2/MatMul/ReadVariableOp26
h3/BiasAdd/ReadVariableOph3/BiasAdd/ReadVariableOp24
h3/MatMul/ReadVariableOph3/MatMul/ReadVariableOp2>
output/BiasAdd/ReadVariableOpoutput/BiasAdd/ReadVariableOp2<
output/MatMul/ReadVariableOpoutput/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
>__inference_h3_layer_call_and_return_conditional_losses_146763

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:

identity_1??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoidb
mulMulBiasAdd:output:0Sigmoid:y:0*
T0*'
_output_shapes
:?????????2
mul[
IdentityIdentitymul:z:0*
T0*'
_output_shapes
:?????????2

Identity?
	IdentityN	IdentityNmul:z:0BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-146756*:
_output_shapes(
&:?????????:?????????2
	IdentityN?

Identity_1IdentityIdentityN:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
>__inference_h0_layer_call_and_return_conditional_losses_145988

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:

identity_1??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoidb
mulMulBiasAdd:output:0Sigmoid:y:0*
T0*'
_output_shapes
:?????????2
mul[
IdentityIdentitymul:z:0*
T0*'
_output_shapes
:?????????2

Identity?
	IdentityN	IdentityNmul:z:0BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-145981*:
_output_shapes(
&:?????????:?????????2
	IdentityN?

Identity_1IdentityIdentityN:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
#__inference_nn_layer_call_fn_146346	
input
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *G
fBR@
>__inference_nn_layer_call_and_return_conditional_losses_1462982
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:N J
'
_output_shapes
:?????????

_user_specified_nameinput
?
F
*__inference_dropout_1_layer_call_fn_146703

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_1462002
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
7
input.
serving_default_input:0?????????:
output0
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?H
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer_with_weights-3
layer-7
	layer-8

layer_with_weights-4

layer-9
	optimizer
loss
trainable_variables
regularization_losses
	variables
	keras_api

signatures
+?&call_and_return_all_conditional_losses
?__call__
?_default_save_signature"?D
_tf_keras_network?D{"name": "nn", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "NN", "config": {"name": "nn", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input"}, "name": "input", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "h0", "trainable": true, "dtype": "float32", "units": 30, "activation": "swish", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "h0", "inbound_nodes": [[["input", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.0, "noise_shape": null, "seed": null}, "name": "dropout", "inbound_nodes": [[["h0", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "h1", "trainable": true, "dtype": "float32", "units": 30, "activation": "swish", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "h1", "inbound_nodes": [[["dropout", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.0, "noise_shape": null, "seed": null}, "name": "dropout_1", "inbound_nodes": [[["h1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "h2", "trainable": true, "dtype": "float32", "units": 30, "activation": "swish", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "h2", "inbound_nodes": [[["dropout_1", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_2", "trainable": true, "dtype": "float32", "rate": 0.0, "noise_shape": null, "seed": null}, "name": "dropout_2", "inbound_nodes": [[["h2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "h3", "trainable": true, "dtype": "float32", "units": 30, "activation": "swish", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "h3", "inbound_nodes": [[["dropout_2", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_3", "trainable": true, "dtype": "float32", "rate": 0.0, "noise_shape": null, "seed": null}, "name": "dropout_3", "inbound_nodes": [[["h3", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "output", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "output", "inbound_nodes": [[["dropout_3", 0, 0, {}]]]}], "input_layers": [["input", 0, 0]], "output_layers": [["output", 0, 0]]}, "shared_object_id": 16, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 2]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 2]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 2]}, "float32", "input"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "NN", "config": {"name": "nn", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input"}, "name": "input", "inbound_nodes": [], "shared_object_id": 0}, {"class_name": "Dense", "config": {"name": "h0", "trainable": true, "dtype": "float32", "units": 30, "activation": "swish", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "h0", "inbound_nodes": [[["input", 0, 0, {}]]], "shared_object_id": 3}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.0, "noise_shape": null, "seed": null}, "name": "dropout", "inbound_nodes": [[["h0", 0, 0, {}]]], "shared_object_id": 4}, {"class_name": "Dense", "config": {"name": "h1", "trainable": true, "dtype": "float32", "units": 30, "activation": "swish", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 5}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "h1", "inbound_nodes": [[["dropout", 0, 0, {}]]], "shared_object_id": 6}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.0, "noise_shape": null, "seed": null}, "name": "dropout_1", "inbound_nodes": [[["h1", 0, 0, {}]]], "shared_object_id": 7}, {"class_name": "Dense", "config": {"name": "h2", "trainable": true, "dtype": "float32", "units": 30, "activation": "swish", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 8}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "h2", "inbound_nodes": [[["dropout_1", 0, 0, {}]]], "shared_object_id": 9}, {"class_name": "Dropout", "config": {"name": "dropout_2", "trainable": true, "dtype": "float32", "rate": 0.0, "noise_shape": null, "seed": null}, "name": "dropout_2", "inbound_nodes": [[["h2", 0, 0, {}]]], "shared_object_id": 10}, {"class_name": "Dense", "config": {"name": "h3", "trainable": true, "dtype": "float32", "units": 30, "activation": "swish", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 11}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "h3", "inbound_nodes": [[["dropout_2", 0, 0, {}]]], "shared_object_id": 12}, {"class_name": "Dropout", "config": {"name": "dropout_3", "trainable": true, "dtype": "float32", "rate": 0.0, "noise_shape": null, "seed": null}, "name": "dropout_3", "inbound_nodes": [[["h3", 0, 0, {}]]], "shared_object_id": 13}, {"class_name": "Dense", "config": {"name": "output", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 14}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "output", "inbound_nodes": [[["dropout_3", 0, 0, {}]]], "shared_object_id": 15}], "input_layers": [["input", 0, 0]], "output_layers": [["output", 0, 0]]}}, "training_config": {"loss": null, "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 9.999999747378752e-05, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input"}}
?

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "h0", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "h0", "trainable": true, "dtype": "float32", "units": 30, "activation": "swish", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["input", 0, 0, {}]]], "shared_object_id": 3, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 2}}, "shared_object_id": 18}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 2]}}
?
trainable_variables
regularization_losses
	variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.0, "noise_shape": null, "seed": null}, "inbound_nodes": [[["h0", 0, 0, {}]]], "shared_object_id": 4}
?

kernel
bias
trainable_variables
regularization_losses
 	variables
!	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "h1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "h1", "trainable": true, "dtype": "float32", "units": 30, "activation": "swish", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 5}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["dropout", 0, 0, {}]]], "shared_object_id": 6, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 30}}, "shared_object_id": 19}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30]}}
?
"trainable_variables
#regularization_losses
$	variables
%	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "dropout_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.0, "noise_shape": null, "seed": null}, "inbound_nodes": [[["h1", 0, 0, {}]]], "shared_object_id": 7}
?

&kernel
'bias
(trainable_variables
)regularization_losses
*	variables
+	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "h2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "h2", "trainable": true, "dtype": "float32", "units": 30, "activation": "swish", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 8}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["dropout_1", 0, 0, {}]]], "shared_object_id": 9, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 30}}, "shared_object_id": 20}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30]}}
?
,trainable_variables
-regularization_losses
.	variables
/	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "dropout_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_2", "trainable": true, "dtype": "float32", "rate": 0.0, "noise_shape": null, "seed": null}, "inbound_nodes": [[["h2", 0, 0, {}]]], "shared_object_id": 10}
?

0kernel
1bias
2trainable_variables
3regularization_losses
4	variables
5	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "h3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "h3", "trainable": true, "dtype": "float32", "units": 30, "activation": "swish", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 11}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["dropout_2", 0, 0, {}]]], "shared_object_id": 12, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 30}}, "shared_object_id": 21}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30]}}
?
6trainable_variables
7regularization_losses
8	variables
9	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "dropout_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_3", "trainable": true, "dtype": "float32", "rate": 0.0, "noise_shape": null, "seed": null}, "inbound_nodes": [[["h3", 0, 0, {}]]], "shared_object_id": 13}
?

:kernel
;bias
<trainable_variables
=regularization_losses
>	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "output", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "output", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 14}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["dropout_3", 0, 0, {}]]], "shared_object_id": 15, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 30}}, "shared_object_id": 22}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30]}}
?
@iter

Abeta_1

Bbeta_2
	Cdecay
Dlearning_ratem|m}m~m&m?'m?0m?1m?:m?;m?v?v?v?v?&v?'v?0v?1v?:v?;v?"
	optimizer
 "
trackable_dict_wrapper
f
0
1
2
3
&4
'5
06
17
:8
;9"
trackable_list_wrapper
 "
trackable_list_wrapper
f
0
1
2
3
&4
'5
06
17
:8
;9"
trackable_list_wrapper
?

Elayers
Fnon_trainable_variables
trainable_variables
regularization_losses
Glayer_metrics
Hlayer_regularization_losses
	variables
Imetrics
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
:2	h0/kernel
:2h0/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?

Jlayers
Knon_trainable_variables
trainable_variables
regularization_losses
Llayer_metrics
Mlayer_regularization_losses
	variables
Nmetrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?

Olayers
Pnon_trainable_variables
trainable_variables
regularization_losses
Qlayer_metrics
Rlayer_regularization_losses
	variables
Smetrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:2	h1/kernel
:2h1/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?

Tlayers
Unon_trainable_variables
trainable_variables
regularization_losses
Vlayer_metrics
Wlayer_regularization_losses
 	variables
Xmetrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?

Ylayers
Znon_trainable_variables
"trainable_variables
#regularization_losses
[layer_metrics
\layer_regularization_losses
$	variables
]metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:2	h2/kernel
:2h2/bias
.
&0
'1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
?

^layers
_non_trainable_variables
(trainable_variables
)regularization_losses
`layer_metrics
alayer_regularization_losses
*	variables
bmetrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?

clayers
dnon_trainable_variables
,trainable_variables
-regularization_losses
elayer_metrics
flayer_regularization_losses
.	variables
gmetrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:2	h3/kernel
:2h3/bias
.
00
11"
trackable_list_wrapper
 "
trackable_list_wrapper
.
00
11"
trackable_list_wrapper
?

hlayers
inon_trainable_variables
2trainable_variables
3regularization_losses
jlayer_metrics
klayer_regularization_losses
4	variables
lmetrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?

mlayers
nnon_trainable_variables
6trainable_variables
7regularization_losses
olayer_metrics
player_regularization_losses
8	variables
qmetrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:2output/kernel
:2output/bias
.
:0
;1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
:0
;1"
trackable_list_wrapper
?

rlayers
snon_trainable_variables
<trainable_variables
=regularization_losses
tlayer_metrics
ulayer_regularization_losses
>	variables
vmetrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
f
0
1
2
3
4
5
6
7
	8

9"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
'
w0"
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
?
	xtotal
	ycount
z	variables
{	keras_api"?
_tf_keras_metric?{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 23}
:  (2total
:  (2count
.
x0
y1"
trackable_list_wrapper
-
z	variables"
_generic_user_object
 :2Adam/h0/kernel/m
:2Adam/h0/bias/m
 :2Adam/h1/kernel/m
:2Adam/h1/bias/m
 :2Adam/h2/kernel/m
:2Adam/h2/bias/m
 :2Adam/h3/kernel/m
:2Adam/h3/bias/m
$:"2Adam/output/kernel/m
:2Adam/output/bias/m
 :2Adam/h0/kernel/v
:2Adam/h0/bias/v
 :2Adam/h1/kernel/v
:2Adam/h1/bias/v
 :2Adam/h2/kernel/v
:2Adam/h2/bias/v
 :2Adam/h3/kernel/v
:2Adam/h3/bias/v
$:"2Adam/output/kernel/v
:2Adam/output/bias/v
?2?
>__inference_nn_layer_call_and_return_conditional_losses_146507
>__inference_nn_layer_call_and_return_conditional_losses_146565
>__inference_nn_layer_call_and_return_conditional_losses_146379
>__inference_nn_layer_call_and_return_conditional_losses_146412?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
#__inference_nn_layer_call_fn_146128
#__inference_nn_layer_call_fn_146590
#__inference_nn_layer_call_fn_146615
#__inference_nn_layer_call_fn_146346?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
!__inference__wrapped_model_145965?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *$?!
?
input?????????
?2?
>__inference_h0_layer_call_and_return_conditional_losses_146631?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
#__inference_h0_layer_call_fn_146640?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dropout_layer_call_and_return_conditional_losses_146645
C__inference_dropout_layer_call_and_return_conditional_losses_146649?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
(__inference_dropout_layer_call_fn_146654
(__inference_dropout_layer_call_fn_146659?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
>__inference_h1_layer_call_and_return_conditional_losses_146675?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
#__inference_h1_layer_call_fn_146684?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_dropout_1_layer_call_and_return_conditional_losses_146689
E__inference_dropout_1_layer_call_and_return_conditional_losses_146693?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
*__inference_dropout_1_layer_call_fn_146698
*__inference_dropout_1_layer_call_fn_146703?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
>__inference_h2_layer_call_and_return_conditional_losses_146719?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
#__inference_h2_layer_call_fn_146728?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_dropout_2_layer_call_and_return_conditional_losses_146733
E__inference_dropout_2_layer_call_and_return_conditional_losses_146737?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
*__inference_dropout_2_layer_call_fn_146742
*__inference_dropout_2_layer_call_fn_146747?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
>__inference_h3_layer_call_and_return_conditional_losses_146763?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
#__inference_h3_layer_call_fn_146772?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_dropout_3_layer_call_and_return_conditional_losses_146777
E__inference_dropout_3_layer_call_and_return_conditional_losses_146781?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
*__inference_dropout_3_layer_call_fn_146786
*__inference_dropout_3_layer_call_fn_146791?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
B__inference_output_layer_call_and_return_conditional_losses_146801?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_output_layer_call_fn_146810?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
$__inference_signature_wrapper_146445input"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
!__inference__wrapped_model_145965m
&'01:;.?+
$?!
?
input?????????
? "/?,
*
output ?
output??????????
E__inference_dropout_1_layer_call_and_return_conditional_losses_146689\3?0
)?&
 ?
inputs?????????
p 
? "%?"
?
0?????????
? ?
E__inference_dropout_1_layer_call_and_return_conditional_losses_146693\3?0
)?&
 ?
inputs?????????
p
? "%?"
?
0?????????
? }
*__inference_dropout_1_layer_call_fn_146698O3?0
)?&
 ?
inputs?????????
p 
? "??????????}
*__inference_dropout_1_layer_call_fn_146703O3?0
)?&
 ?
inputs?????????
p
? "???????????
E__inference_dropout_2_layer_call_and_return_conditional_losses_146733\3?0
)?&
 ?
inputs?????????
p 
? "%?"
?
0?????????
? ?
E__inference_dropout_2_layer_call_and_return_conditional_losses_146737\3?0
)?&
 ?
inputs?????????
p
? "%?"
?
0?????????
? }
*__inference_dropout_2_layer_call_fn_146742O3?0
)?&
 ?
inputs?????????
p 
? "??????????}
*__inference_dropout_2_layer_call_fn_146747O3?0
)?&
 ?
inputs?????????
p
? "???????????
E__inference_dropout_3_layer_call_and_return_conditional_losses_146777\3?0
)?&
 ?
inputs?????????
p 
? "%?"
?
0?????????
? ?
E__inference_dropout_3_layer_call_and_return_conditional_losses_146781\3?0
)?&
 ?
inputs?????????
p
? "%?"
?
0?????????
? }
*__inference_dropout_3_layer_call_fn_146786O3?0
)?&
 ?
inputs?????????
p 
? "??????????}
*__inference_dropout_3_layer_call_fn_146791O3?0
)?&
 ?
inputs?????????
p
? "???????????
C__inference_dropout_layer_call_and_return_conditional_losses_146645\3?0
)?&
 ?
inputs?????????
p 
? "%?"
?
0?????????
? ?
C__inference_dropout_layer_call_and_return_conditional_losses_146649\3?0
)?&
 ?
inputs?????????
p
? "%?"
?
0?????????
? {
(__inference_dropout_layer_call_fn_146654O3?0
)?&
 ?
inputs?????????
p 
? "??????????{
(__inference_dropout_layer_call_fn_146659O3?0
)?&
 ?
inputs?????????
p
? "???????????
>__inference_h0_layer_call_and_return_conditional_losses_146631\/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? v
#__inference_h0_layer_call_fn_146640O/?,
%?"
 ?
inputs?????????
? "???????????
>__inference_h1_layer_call_and_return_conditional_losses_146675\/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? v
#__inference_h1_layer_call_fn_146684O/?,
%?"
 ?
inputs?????????
? "???????????
>__inference_h2_layer_call_and_return_conditional_losses_146719\&'/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? v
#__inference_h2_layer_call_fn_146728O&'/?,
%?"
 ?
inputs?????????
? "???????????
>__inference_h3_layer_call_and_return_conditional_losses_146763\01/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? v
#__inference_h3_layer_call_fn_146772O01/?,
%?"
 ?
inputs?????????
? "???????????
>__inference_nn_layer_call_and_return_conditional_losses_146379k
&'01:;6?3
,?)
?
input?????????
p 

 
? "%?"
?
0?????????
? ?
>__inference_nn_layer_call_and_return_conditional_losses_146412k
&'01:;6?3
,?)
?
input?????????
p

 
? "%?"
?
0?????????
? ?
>__inference_nn_layer_call_and_return_conditional_losses_146507l
&'01:;7?4
-?*
 ?
inputs?????????
p 

 
? "%?"
?
0?????????
? ?
>__inference_nn_layer_call_and_return_conditional_losses_146565l
&'01:;7?4
-?*
 ?
inputs?????????
p

 
? "%?"
?
0?????????
? ?
#__inference_nn_layer_call_fn_146128^
&'01:;6?3
,?)
?
input?????????
p 

 
? "???????????
#__inference_nn_layer_call_fn_146346^
&'01:;6?3
,?)
?
input?????????
p

 
? "???????????
#__inference_nn_layer_call_fn_146590_
&'01:;7?4
-?*
 ?
inputs?????????
p 

 
? "???????????
#__inference_nn_layer_call_fn_146615_
&'01:;7?4
-?*
 ?
inputs?????????
p

 
? "???????????
B__inference_output_layer_call_and_return_conditional_losses_146801\:;/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? z
'__inference_output_layer_call_fn_146810O:;/?,
%?"
 ?
inputs?????????
? "???????????
$__inference_signature_wrapper_146445v
&'01:;7?4
? 
-?*
(
input?
input?????????"/?,
*
output ?
output?????????