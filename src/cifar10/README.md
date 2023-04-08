`decomposition.py`: Provide attack decomposition on CIFAR-10 dataset.

There are two options for argument **--func**, 'mask' and 'transform', where 'mask' denotes using patch function to extract the trigger
and 'transform' denotes using transforming function to approximate the trigger effect.

Moreover, there are two options for argument **--func_option**, denoting two parameter distributions for both function, e.g., 'binomial' and 'uniform' for patch function
and 'simple' and 'complex' for transforming function.

For example, patch function with binomial distribution is good at handling BadNets trigger, patch function with uniform distribution is suitable for Refool trigger,
and transforming function with complex distribution works well on WaNet trigger.

`python decomposition.py --dataset cifar10 --network vgg11 --attack badnet --func mask --func_option binomial --epochs 1000`

`python decomposition.py --dataset cifar10 --network vgg11 --attack refool_smooth --func mask --func_option uniform --epochs 1000`

`python decomposition.py --dataset cifar10 --network vgg11 --attack wanet --func transform --func_option complex --epochs 1000`

======================================================================================

Besides we also provide enhanced ABS to deal with patch triggers and complex triggers.

`abs_beagle_mask_binomial.py`

`abs_beagle_transform_complex.py`
