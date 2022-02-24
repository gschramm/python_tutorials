; example on how to call the parallelized versions of the finite difference gradient / divergence
; implemented in pymirc from IDL

np = Python.Import('numpy')
pi = Python.Import('pymirc.image_operations')

; dimension of 3D image
shape = [300,300,150]
seed  = 1

; setup random 2D/3D/4D image
x = randomu(seed, shape)

; allocate array for gradient of image
print, 'parallel gradient'
TIC
grad_x = fltarr([x.dim, x.ndim]) 
TOC
; calculate the gradient of the image
tmp    = pi.grad(x, grad_x) 

; setup random gradient image
y = randomu(seed, [x.dim, x.ndim])
; calculate divergence
print, 'parallel divergence'
TIC
div_y = pi.div(y)
TOC

; singe CPU version of gradient
; note that in numpy the axis order is reversed (right most index raising the fastest)

naive_grad = fltarr(grad_x.dim)
print, 'naive gradient'
TIC
naive_grad[0:-2,*,*,2] = x[1:-1,*,*] - x[0:-2,*,*]
naive_grad[*,0:-2,*,1] = x[*,1:-1,*] - x[*,0:-2,*]
naive_grad[*,*,0:-2,0] = x[*,*,1:-1] - x[*,*,0:-2]
TOC

; single CPU divergence
naive_div = fltarr(x.dim)

print, 'naive divergence'
TIC
naive_div[1:-1,*,*] += (y[1:-1,*,*,2] - y[0:-2,*,*,2])
naive_div[*,1:-1,*] += (y[*,1:-1,*,1] - y[*,0:-2,*,1])
naive_div[*,*,1:-1] += (y[*,*,1:-1,0] - y[*,*,0:-2,0])
TOC

; check if parallel gradient and naive gradient are the same
; gradient should be exactly the same, the divergence shoul dbe within floating point precision

print, max(abs(grad_x - naive_grad))
print, max(abs(naive_div - div_y))

END
