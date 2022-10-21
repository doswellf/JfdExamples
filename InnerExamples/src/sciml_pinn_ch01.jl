#
# Scientific Machine Learning Through Physics-Informed Neural Networks
#
using Flux

W = [randn(32,10),randn(32,32),randn(5,32)]
b = [zeros(32),zeros(32),zeros(5)]

simpleNN(x) = W[3] * tanh.(W[2]*tanh.(W[1] * x + b[1]) + b[2]) + b[3]
simpleNN(rand(10))

NN2 = Chain(
          Dense(10,32,tanh)
        , Dense(32,32,tanh)
        , Dense(32,5)
)

NN2(rand(10))
NN2[1].weight
# Using an anonymous function to create a custom activation function (quadratic)

NN3 = Chain(
      Dense(10,32, x -> x^2)
    , Dense(32,32, x -> max(0,x))
    , Dense(32,5)
)

NN3(rand(10))
NN3[1].weight
# looking under the hood
# NN2 and NN3 are just functions with the following form
# simpleNN(x) = W[3] * tanh.(W[2] * tanh.(W[1]*x + b[1]) + b[2]) + b[3]
#

using InteractiveUtils

@which Dense(10,32,tanh)

f = Dense(32,32,tanh)
f(rand(32))

@which Chain(1,2,3)

slurper(xs...) = @show xs
slurper(1,2,3,4,5)

# training the neural network - find weights that minimize the loss function

# trivial function returning 1 for any input x ⋿ [0,1] of size 10

NNT01 = Chain(
    Dense(10,32,tanh)
  , Dense(32,32,tanh)
  , Dense(32,5)
)


loss() = sum(abs2, sum(abs2, NNT01(rand(10)).-1) for i in 1:100)
loss()

# The weight matrix of the first layer
# This appears to be wrong or out of date
# NNT01[1].W
#
# use NNT01[1].weight NNT01[2].weight, etc
#
# Also out of date!
# p = params(NNT01)
# use p = Flux.params(NNT01)
#

NNT01[1].weight
NNT01[2].weight
NNT01[3].weight

p_all = Flux.params(NNT01)
p1 = Flux.params(NNT01[1])
p2 = Flux.params(NNT01[2])
p3 = Flux.params(NNT01[3])

#=
  Now let's get into training neural networks.
  "Training" a neural network is simply the process of finding weights
  that minimize a loss function.
  For example, let's say we wanted to make our neural network be the
  constant function 1 for any input x∈[0,1]10.
  We can then write the loss function as follows:
=#

NNT02 = Chain(
      Dense(10,32,tanh)
    , Dense(32,32,tanh)
    , Dense(32,5)
)

loss() = sum(abs2, sum(abs2, NNT02(rand(10)).-1) for i in 1:100)
loss()

NNT02[1].weight
p_nnt02 = Flux.params(NNT02)

# Find optimal parameter values p that cause NNT02 to become the constant 1  function

Flux.train!(loss,p_nnt02,Iterators.repeated((), 10000), Adam(0.1))

loss()

# First true SciML application
# Solve ODE's with neural networks
# Also known as using a DE as a regularizer in the loss function
# This is a physics-informed neural network
#
# Use u' = cos 2πt and approximate it with the NN
# Takes a scalar and returns a scalar
#
using Flux
NNODE = Chain(
    x -> [x]        # transform a scalar into an array
  , Dense(1,32,tanh)
  , Dense(32,1)
  ,first            # take the first value, i.e. return a scalar
)

NNODE(1.0)

# use the transformed eqn  forced to satisfy the boundary condition
# use u0 = 1.0 gives the universal approximator:

g(t) = t * NNODE(t) + 1f0

# for g to be a fn that satisfies g' = cos 2πt we need loss below to be minimized

using Statistics
ϵ = sqrt(eps(Float32))
loss() = mean(abs2(((g(t+ϵ) - g(t))/ϵ) - cos(2π*t)) for t in 0:1f-2:1f0)

opt = Flux.Descent(0.01)
data = Iterators.repeated((),5000)
iter = 0
cb = function ()  # callback fn to observe training
      global iter += 1
      if iter % 500 == 0
        display(loss())
      end
    end

  display(loss())
  Flux.train!(loss,Flux.params(NNODE), data, opt; cb=cb)

# integrate both sides of ODE to get C + (sin 2πt/2π)
# where C = 1

# use (input,output) pairs from the NN and plot vs the analytical solution

using Plots
t = 0:0.001:1.0
plot(t,g.(t),labels="NNODE")
plot!(t,1.0 .+ sin.(2π.*t)/2π, label = "True Solution")

# Using Physics-Informed NN For Hookes Spring Law

# Using Physics-Informed NN For Harmonic Oscillator

using DifferentialEquations
k =  1.0
force(dx,x,k,t) = -k * x+ 0.1sin(x)
prob = SecondOrderODEProblem(force,1.0,0.0,(0.0,10.0),k)
sol = solve(prob)
plot(sol,label=["Velocity","Position"])

plot_t = 0:0.01:10
data_plot = sol(plot_t)
positions_plot = [state[2] for state in data_plot]
force_plot = [force(state[1],state[2],k,t) for state in data_plot]

# generate the dataset
t = 0:3.3:10
dataset=sol(t)
position_data = [state[2] for state in sol(t)]
force_data = [force(state[1],state[2],k,t) for state in sol(t)]

plot(plot_t,force_plot,xlabel="t",label="True Force")
scatter!(t,force_data,label="Force Measurements")
W = [randn(32,10),randn(32,32),randn(5,32)]
b = [zeros(32),zeros(32),zeros(5)]

simpleNN(x) = W[3] * tanh.(W[2]*tanh.(W[1] * x + b[1]) + b[2]) + b[3]
simpleNN(rand(10))

NN2 = Chain(
          Dense(10,32,tanh)
        , Dense(32,32,tanh)
        , Dense(32,5)
)

NN2(rand(10))
NN2[1].weight
# Using an anonymous function to create a custom activation function (quadratic)

NN3 = Chain(
      Dense(10,32, x -> x^2)
    , Dense(32,32, x -> max(0,x))
    , Dense(32,5)
)

NN3(rand(10))
NN3[1].weight
# looking under the hood
# NN2 and NN3 are just functions with the following form
# simpleNN(x) = W[3] * tanh.(W[2] * tanh.(W[1]*x + b[1]) + b[2]) + b[3]
#

using InteractiveUtils

@which Dense(10,32,tanh)

f = Dense(32,32,tanh)
f(rand(32))

@which Chain(1,2,3)

slurper(xs...) = @show xs
slurper(1,2,3,4,5)

# training the neural network - find weights that minimize the loss function

# trivial function returning 1 for any input x ⋿ [0,1] of size 10

NNT01 = Chain(
    Dense(10,32,tanh)
  , Dense(32,32,tanh)
  , Dense(32,5)
)

loss() = sum(abs2, sum(abs2, NNT01(rand(10)).-1) for i in 1:100)
loss()

# The weight matrix of the first layer
# This appears to be wrong or out of date
# NNT01[1].W
#
# use NNT01[1].weight NNT01[2].weight, etc
#
# Also out of date!
# p = params(NNT01)
# use p = Flux.params(NNT01)
#

NNT01[1].weight
NNT01[2].weight
NNT01[3].weight

p_all = Flux.params(NNT01)
p1 = Flux.params(NNT01[1])
p2 = Flux.params(NNT01[2])
p3 = Flux.params(NNT01[3])

#=
  Now let's get into training neural networks.
  "Training" a neural network is simply the process of finding weights
  that minimize a loss function.
  For example, let's say we wanted to make our neural network be the
  constant function 1 for any input x∈[0,1]10.
  We can then write the loss function as follows:
=#

NNT02 = Chain(
      Dense(10,32,tanh)
    , Dense(32,32,tanh)
    , Dense(32,5)
)

loss() = sum(abs2, sum(abs2, NNT02(rand(10)).-1) for i in 1:100)
loss()

NNT02[1].weight
p_nnt02 = Flux.params(NNT02)

# Find optimal parameter values p that cause NNT02 to become the constant 1  function

Flux.train!(loss,p_nnt02,Iterators.repeated((), 10000), Adam(0.1))

loss()

# First true SciML application
# Solve ODE's with neural networks
# Also known as using a DE as a regularizer in the loss function
# This is a physics-informed neural network
#
# Use u' = cos 2πt and approximate it with the NN
# Takes a scalar and returns a scalar
#
using Flux
NNODE = Chain(
    x -> [x]        # transform a scalar into an array
  , Dense(1,32,tanh)
  , Dense(32,1)
  ,first            # take the first value, i.e. return a scalar
)

NNODE(1.0)

# use the transformed eqn  forced to satisfy the boundary condition
# use u0 = 1.0 gives the universal approximator:

g(t) = t * NNODE(t) + 1f0

# for g to be a fn that satisfies g' = cos 2πt we need loss below to be minimized

using Statistics
ϵ = sqrt(eps(Float32))
loss() = mean(abs2(((g(t+ϵ) - g(t))/ϵ) - cos(2π*t)) for t in 0:1f-2:1f0)

opt = Flux.Descent(0.01)
data = Iterators.repeated((),5000)
iter = 0
cb = function ()  # callback fn to observe training
      global iter += 1
      if iter % 500 == 0
        display(loss())
      end
    end

  display(loss())
  Flux.train!(loss,Flux.params(NNODE), data, opt; cb=cb)

# integrate both sides of ODE to get C + (sin 2πt/2π)
# where C = 1

# use (input,output) pairs from the NN and plot vs the analytical solution

using Plots
t = 0:0.001:1.0
plot(t,g.(t),labels="NNODE")
plot!(t,1.0 .+ sin.(2π.*t)/2π, label = "True Solution")

# Using Physics-Informed NN For Hookes Spring Law

# Using Physics-Informed NN For Harmonic Oscillator

using DifferentialEquations
k =  1.0
force(dx,x,k,t) = -k * x+ 0.1sin(x)
prob = SecondOrderODEProblem(force,1.0,0.0,(0.0,10.0),k)
sol = solve(prob)
plot(sol,label=["Velocity","Position"])

plot_t = 0:0.01:10
W = [randn(32,10),randn(32,32),randn(5,32)]
b = [zeros(32),zeros(32),zeros(5)]

simpleNN(x) = W[3] * tanh.(W[2]*tanh.(W[1] * x + b[1]) + b[2]) + b[3]
simpleNN(rand(10))

NN2 = Chain(
          Dense(10,32,tanh)
        , Dense(32,32,tanh)
        , Dense(32,5)
)

NN2(rand(10))
NN2[1].weight
# Using an anonymous function to create a custom activation function (quadratic)

NN3 = Chain(
      Dense(10,32, x -> x^2)
    , Dense(32,32, x -> max(0,x))
    , Dense(32,5)
)

NN3(rand(10))
NN3[1].weight
# looking under the hood
# NN2 and NN3 are just functions with the following form
# simpleNN(x) = W[3] * tanh.(W[2] * tanh.(W[1]*x + b[1]) + b[2]) + b[3]
#

using InteractiveUtils

@which Dense(10,32,tanh)

f = Dense(32,32,tanh)
f(rand(32))

@which Chain(1,2,3)

slurper(xs...) = @show xs
slurper(1,2,3,4,5)

# training the neural network - find weights that minimize the loss function

# trivial function returning 1 for any input x ⋿ [0,1] of size 10

NNT01 = Chain(
    Dense(10,32,tanh)
  , Dense(32,32,tanh)
  , Dense(32,5)
)

loss() = sum(abs2, sum(abs2, NNT01(rand(10)).-1) for i in 1:100)
loss()

# The weight matrix of the first layer
# This appears to be wrong or out of date
# NNT01[1].W
#
# use NNT01[1].weight NNT01[2].weight, etc
#
# Also out of date!
# p = params(NNT01)
# use p = Flux.params(NNT01)
#

NNT01[1].weight
NNT01[2].weight
NNT01[3].weight

p_all = Flux.params(NNT01)
p1 = Flux.params(NNT01[1])
p2 = Flux.params(NNT01[2])
p3 = Flux.params(NNT01[3])

#=
  Now let's get into training neural networks.
  "Training" a neural network is simply the process of finding weights
  that minimize a loss function.
  For example, let's say we wanted to make our neural network be the
  constant function 1 for any input x∈[0,1]10.
  We can then write the loss function as follows:
=#

NNT02 = Chain(
      Dense(10,32,tanh)
    , Dense(32,32,tanh)
    , Dense(32,5)
)

loss() = sum(abs2, sum(abs2, NNT02(rand(10)).-1) for i in 1:100)
loss()

NNT02[1].weight
p_nnt02 = Flux.params(NNT02)

# Find optimal parameter values p that cause NNT02 to become the constant 1  function

Flux.train!(loss,p_nnt02,Iterators.repeated((), 10000), Adam(0.1))

loss()

# First true SciML application
# Solve ODE's with neural networks
# Also known as using a DE as a regularizer in the loss function
# This is a physics-informed neural network
#
# Use u' = cos 2πt and approximate it with the NN
# Takes a scalar and returns a scalar
#
using Flux
NNODE = Chain(
    x -> [x]        # transform a scalar into an array
  , Dense(1,32,tanh)
  , Dense(32,1)
  ,first            # take the first value, i.e. return a scalar
)

NNODE(1.0)

# use the transformed eqn  forced to satisfy the boundary condition
# use u0 = 1.0 gives the universal approximator:

g(t) = t * NNODE(t) + 1f0

# for g to be a fn that satisfies g' = cos 2πt we need loss below to be minimized

using Statistics
ϵ = sqrt(eps(Float32))
loss() = mean(abs2(((g(t+ϵ) - g(t))/ϵ) - cos(2π*t)) for t in 0:1f-2:1f0)

opt = Flux.Descent(0.01)
data = Iterators.repeated((),5000)
iter = 0
cb = function ()  # callback fn to observe training
      global iter += 1
      if iter % 500 == 0
        display(loss())
      end
    end

  display(loss())
  Flux.train!(loss,Flux.params(NNODE), data, opt; cb=cb)

# integrate both sides of ODE to get C + (sin 2πt/2π)
# where C = 1

# use (input,output) pairs from the NN and plot vs the analytical solution

using Plots
t = 0:0.001:1.0
plot(t,g.(t),labels="NNODE")
plot!(t,1.0 .+ sin.(2π.*t)/2π, label = "True Solution")

# Using Physics-Informed NN For Hookes Spring Law

# Using Physics-Informed NN For Harmonic Oscillator

using DifferentialEquations
k =  1.0
force(dx,x,k,t) = -k * x+ 0.1sin(x)
prob = SecondOrderODEProblem(force,1.0,0.0,(0.0,10.0),k)
sol = solve(prob)
plot(sol,label=["Velocity","Position"])

plot_t = 0:0.01:10
data_plot = sol(plot_t)
positions_plot = [state[2] for state in data_plot]
force_plot = [force(state[1],state[2],k,t) for state in data_plot]

# generate the dataset
t = 0:3.3:10
dataset=sol(t)
position_data = [state[2] for state in sol(t)]
force_data = [force(state[1],state[2],k,t) for state in sol(t)]

plot(plot_t,force_plot,xlabel="t",label="True Force")
scatter!(t,force_data,label="Force Measurements")

NNForce = Chain(
    x -> [x]
  , Dense(1,32,tanh)
  , Dense(32,1)
  , first
)

loss() = sum(abs2,NNForce(position_data[i])
  - force_data[i] for i in 1:length(position_data))
  loss()

  # random parameters do not work as well, time to train

  opt = Flux.Descent(0.01)
  data = Iterators.repeated((),5000)
  iter = 0

  cb = function ()  # callback function
    global iter += 1
    if iter % 500 == 0
      display(loss())
    end
  end

display(loss())

Flux.train!(loss,Flux.params(NNForce), data, opt; cb=cb)

data_plot = sol(plot_t)
positions_plot = [state[2] for state in data_plot]
force_plot = [force(state[1],state[2],k,t) for state in data_plot]

# generate the dataset
t = 0:3.3:10
dataset=sol(t)
position_data = [state[2] for state in sol(t)]
force_data = [force(state[1],state[2],k,t) for state in sol(t)]

plot(plot_t,force_plot,xlabel="t",label="True Force")
scatter!(t,force_data,label="Force Measurements")

NNForce = Chain(
    x -> [x]
  , Dense(1,32,tanh)
  , Dense(32,1)
  , first
)

loss() = sum(abs2,NNForce(position_data[i])
  - force_data[i] for i in 1:length(position_data))
  loss()

  # random parameters do not work as well, time to train

  opt = Flux.Descent(0.01)
  data = Iterators.repeated((),5000)
  iter = 0

  cb = function ()  # callback function
    global iter += 1
    if iter % 500 == 0
      display(loss())
    end
  end
  display(loss())
  Flux.train!(loss,Flux.params(NNForce), data, opt; cb=cb)

  learned_force_plot = NNForce.(positions_plot)

  plot(plot_t,force_plot,xlabel="t",label="True Force")
  plot!(plot_t,learned_force_plot,label="Predicted Force")
  scatter!(t,force_data,label="Force Measurements")

NNForce = Chain(
    x -> [x]
  , Dense(1,32,tanh)
  , Dense(32,1)
  , first
)

loss() = sum(abs2,NNForce(position_data[i])
  - force_data[i] for i in 1:length(position_data))
loss()

  # random parameters do not work as well, time to train

  opt = Flux.Descent(0.01)
  data = Iterators.repeated((),5000)
  iter = 0

  cb = function ()  # callback function
    global iter += 1
    if iter % 500 == 0
      display(loss())
    end
  end

display(loss())
Flux.train!(loss,Flux.params(NNForce), data, opt; cb=cb)
learned_force_plot = NNForce.(positions_plot)

plot(plot_t,force_plot,xlabel="t",label="True Force")
plot!(plot_t,learned_force_plot,label="Predicted Force")
scatter!(t,force_data,label = "Force Measurements")

# Approximation of the wrong function but it fits the data points
# Use Hooke's law to guide the system F(x) = -kx

force2(dx,x,j,t) = -k*x
prob_simplified = SecondOrderODEProblem(force2,1.0,0.0,(0.0,10.0),k)
sol_simplified = solve(prob_simplified)
plot(sol,label=["Velocity" "Position"])
plot!(sol_simplified, label=["Velocity Simplified" "Position Simplified"] )
random_positions = [2rand()-1 for i in 1:100]
loss_ode() = sum(abs2,NNForce(x) - (-k*x) for x in random_positions)
loss_ode()
λ = 0.1
composed_loss() = loss() + λ*loss_ode()
# λ is a weight factor to control the regularization vs the physics assumption
# now the physics-informed NN can be trained
opt = Flux.Descent(0.01)
data = Iterators.repeated((),5000)
iter = 0
cb = function ()
  global iter += 1
  if iter % 500 == 0
    display(composed_loss())
  end
end
display(composed_loss())
Flux.train!(composed_loss, Flux.params(NNForce) , data, opt; cb=cb)

learned_force_plot = NNForce.(positions_plot)

plot(plot_t, force_plot, xlabel="t", label = "True Force")
plot!(plot_t, learned_force_plot, label="Predicted Force")
scatter!(t, force_data, label = "Force Measurements")

random_positions = [2rand()-1 for i in 1:100] #randome values
loss_ode() = sum(abs2,NNForce(x) - (-k*x) for x in random_positions)
loss_ode()

λ = 0.1
composed_loss() = loss() + λ*loss_ode()

opt = Flux.Descent(0.01)
data = Iterators.repeated((),5000)
iter = 0
cb = function ()
  global iter += 1
  if iter % 500 == 0
    display(composed_loss())
  end
end
display(composed_loss())
Flux.train!(composed_loss,Flux.params(NNForce), data, opt; cb=cb)

learned_force_plot = NNForce.(positions_plot)

plot(plot_t, force_plot, xlabel = "t", label="True Force")
plot!(plot_t,learned_force_plot,label="Predicted Force")
scatter!(t,force_data,label="Force Measurements")
