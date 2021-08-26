### Pseudo 3D Meninges Surface Reconstruction

##### Train

```bash
train.py -t $training_data -o $output_dir -V 1 -v $validation_data
```

`-v` and `-V` (for validation) are optional.

##### Test

```bash
test.py -1 $t1w -w $t2w -o $output_dir -c $checkponit_filename -u
```

`-u` (use CUDA) is optional.
