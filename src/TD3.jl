module TD3

using Flux, Flux.Optimise
import Flux.params
using Distributions
using Statistics
using Conda
using PyCall
using Parameters 
using UnPack
using MLUtils
# import Base.push!



export agent, 
        HyperParameter,
        renderEnv,
        setCritic,
        setActor,
        ReplayBuffer,
        remember,
        sample,
        soft_update!,
        train_step!



@with_kw mutable struct EnvParameter
    # Dimensions
    ## Actions
    action_size::Int =                      1
    action_bound::Float32 =                 1.f0
    action_bound_high::Array{Float32} =     [1.f0]
    action_bound_low::Array{Float32} =      [-1.f0]
    ## States
    state_size::Int =                       1
    state_bound_high::Array{Float32} =      [1.f0]
    state_bound_low::Array{Float32} =       [1.f0]
end

@with_kw mutable struct HyperParameter
    # Buffer size
    buffer_size::Int =                      1000000
    # Exploration
    expl_noise::Float32 =                   0.1f0
    target_noise::Float32 =                 0.2f0
    noise_clip::Float32 =                   0.5f0
    # Training Metrics
    training_episodes::Int =                300
    maximum_episode_length::Int =           3000
    train_start:: Int =                     10
    batch_size::Int =                       64
    policy_delay::Int =                     2  # Add policy update delay
    # Metrics
    episode_reward::Array{Float32} =        []
    critic_loss::Array{Float32} =           [0.f0]
    actor_loss::Array{Float32} =            [0.f0]
    episode_steps::Array{Int} =             []
    # Discount
    γ::Float32 =                            0.99f0
    # Learning Rates
    critic_η::Float64 =                     0.001
    actor_η::Float64 =                      0.001
    # Agents
    store_frequency::Int =                  100
    trained_agents =                        []
end


function setCritic(state_size, action_size)

    critic1 = Chain(Dense(state_size + action_size, 400, relu),
                    Dense(400, 300, relu),
                    Dense(300, 1))

    critic2 = Chain(Dense(state_size + action_size, 400, relu),
                    Dense(400, 300, relu),
                    Dense(300, 1))

    return critic1, critic2

    # return Chain(Dense(state_size + action_size, 4, relu),
    #                 Dense(4, 3, relu),
    #                 Dense(3, 1))
                    
end


function setActor(state_size, action_size)

    # return Chain(Dense(state_size, 4, relu),
    #                 Dense(4, 3, relu),
    #                 Dense(3, action_size, tanh))
    return Chain(Dense(state_size, 400, relu),
                    Dense(400, 300, relu),
                    Dense(300, action_size, tanh))

end



# Define the experience replay buffer
mutable struct ReplayBuffer
    capacity::Int
    memory::Vector{Tuple{Vector{Float32}, Vector{Float32}, Float32, Vector{Float32}, Bool}}
    pos::Int
end

# outer constructor for the Replay Buffer
function ReplayBuffer(capacity::Int)
    memory = []
    return ReplayBuffer(capacity, memory, 1)
end


function remember(buffer::ReplayBuffer, state, action, reward, next_state, done)
    if length(buffer.memory) < buffer.capacity
        push!(buffer.memory, (state, action, reward, next_state, done))
    else
        buffer.memory[buffer.pos] = (state, action, reward, next_state, done)
    end
    buffer.pos = mod1(buffer.pos + 1, buffer.capacity)
end


function sample(buffer::ReplayBuffer, batch_size::Int)
    batch = rand(buffer.memory, batch_size)
    states, actions, rewards, next_states, dones = [], [], [], [], []
    for (s, a, r, ns, d) in batch
        push!(states, s)
        push!(actions, a)
        push!(rewards, r)
        push!(next_states, ns)
        push!(dones, d)
    end
    return hcat(states...), hcat(actions...), rewards, hcat(next_states...), dones
end


# Define the action, actor_loss, and critic_loss functions
function action(model, state, train, ep, hp)
    if train
        a = model(state) .+ clamp.(rand(Normal{Float32}(0.f0, hp.expl_noise), size(ep.action_size)), -hp.noise_clip, hp.noise_clip)
        return clamp.(a, ep.action_bound_low, ep.action_bound_high)
    else
        return model(state)
    end
end



function soft_update!(target_model, main_model, τ)
    for (target_param, main_param) in zip(Flux.params(target_model), Flux.params(main_model))
        target_param .= τ * main_param .+ (1 - τ) * target_param
    end
end



function verify_update(target_model, main_model)
    for (i, (target_param, main_param)) in enumerate(zip(Flux.params(target_model), Flux.params(main_model)))
        diff = main_param - target_param
        println("Difference for parameter $i:")
        println(diff)
    end
end



function train_step!(S, A, R, S´, T, μθ, μθ´, Qϕ1, Qϕ1´, Qϕ2, Qϕ2´, ep::EnvParameter, hp::HyperParameter, step::Int)


    # Add target action noise to the target action
    target_action_noise = clamp.(rand(Normal{Float32}(0.f0, hp.target_noise), size(A)), -hp.noise_clip, hp.noise_clip)
    target_action = μθ´(S) .+ target_action_noise
    
    # Clip target action
    target_action = clamp.(target_action, ep.action_bound_low, ep.action_bound_high)


    Y = R' .+ hp.γ * (1 .- T)' .* min.(Qϕ1´(vcat(S´, target_action)), Qϕ2´(vcat(S´, target_action)))
    # Y = R' .+ hp.γ * (1 .- T)' .* min.(Qϕ1´(vcat(S´, μθ´(S))), Qϕ2´(vcat(S´, μθ´(S))))

    #@show sum(Flux.Losses.mse(Qϕ1(vcat(S, A)), Y) + Flux.Losses.mse(Qϕ2(vcat(S, A)), Y))
    # dϕ1 = Flux.gradient(m -> sum(Flux.Losses.mse(m(vcat(S, A)), Y) + Flux.Losses.mse(Qϕ2(vcat(S, A)), Y)), Qϕ1)

    # Train both critic networks
    dϕ1 = Flux.gradient(m -> Flux.Losses.mse(m(vcat(S, A)), Y), Qϕ1)
    Flux.update!(opt_critic1, Qϕ1, dϕ1[1])
    
    dϕ2 = Flux.gradient(m -> Flux.Losses.mse(m(vcat(S, A)), Y), Qϕ2)
    Flux.update!(opt_critic2, Qϕ2, dϕ2[1])

    
    
    if step % hp.policy_delay == 0
        #actor
        dθ = Flux.gradient(m -> -mean(Qϕ1(vcat(S, m(S)))), μθ)
        Flux.update!(opt_actor, μθ, dθ[1])
        
        # Soft update target networks
        soft_update!(Qϕ1´, Qϕ1, 0.005)
        soft_update!(Qϕ2´, Qϕ2, 0.005)
        soft_update!(μθ´, μθ, 0.005)
    end
    
    # push!(hyperParams.critic_loss, Flux.Losses.mse(Qϕ1(vcat(S, A)), Y))
    # push!(hyperParams.actor_loss, -mean(Qϕ1(vcat(S, μθ(S)))))
    
    #verify_update(Qϕ´, Qϕ)
    
end


function agent(environment, hyperParams::HyperParameter)
    println("Hello people I am here")
    
    gym = pyimport("gym")
    
    if environment == "LunarLander-v2"
        global env = gym.make(environment, continuous=true)
    else
        global env = gym.make(environment)
    end
    
    envParams = EnvParameter()
    
    # Reset Parameters
    ## ActionenvP
    envParams.action_size =        env.action_space.shape[1]
    envParams.action_bound =       env.action_space.high[1]
    envParams.action_bound_high =  env.action_space.high
    envParams.action_bound_low =   env.action_space.low
    ## States
    envParams.state_size =         env.observation_space.shape[1]
    envParams.state_bound_high =   env.observation_space.high
    envParams.state_bound_low =    env.observation_space.low
    
    
    episode = 1
    
    μθ = setActor(envParams.state_size, envParams.action_size)
    μθ´= deepcopy(μθ)
    Qϕ1, Qϕ2 = setCritic(envParams.state_size, envParams.action_size)
    Qϕ1´, Qϕ2´ = deepcopy(Qϕ1), deepcopy(Qϕ2)
    
    global opt_critic1 = Flux.setup(Flux.Optimise.Adam(hyperParams.critic_η), Qϕ1)
    global opt_critic2 = Flux.setup(Flux.Optimise.Adam(hyperParams.critic_η), Qϕ2)
    global opt_actor = Flux.setup(Flux.Optimise.Adam(hyperParams.actor_η), μθ)
    
    buffer = ReplayBuffer(hyperParams.buffer_size)
    
    while episode ≤ hyperParams.training_episodes
        
        frames = 0
        s, info = env.reset()
        episode_rewards = 0
        t = false
        
        for step in 1:hyperParams.maximum_episode_length
            
            a = action(μθ, s, true, envParams, hyperParams)
            s´, r, terminated, truncated, _ = env.step(a)
            
            terminated | truncated ? t = true : t = false
            
            episode_rewards += r
            
            remember(buffer, s, a, r, s´, t)
            
            if episode > hyperParams.train_start
                
                S, A, R, S´, T = sample(buffer, hyperParams.batch_size)
                train_step!(S, A, R, S´, T, μθ, μθ´, Qϕ1, Qϕ1´, Qϕ2, Qϕ2´, envParams, hyperParams, step)
                
            end
            
            
            s = s´
            frames += 1
            
            if t
                env.close()
                break
            end
            
        end
        

        if episode % hyperParams.store_frequency == 0
            push!(hyperParams.trained_agents, deepcopy(μθ))
        end
        


        push!(hyperParams.episode_steps, frames)
        push!(hyperParams.episode_reward, episode_rewards)
        
        println("Episode: $episode | Cumulative Reward: $(round(episode_rewards, digits=2)) | Critic Loss: $(hyperParams.critic_loss[end]) | Actor Loss: $(hyperParams.actor_loss[end]) | Steps: $(frames)")
        episode += 1
    end
    
    return hyperParams
    
end

# Works
# hp = agent("Pendulum-v1", HyperParameter(expl_noise=0.2f0, training_episodes=300, maximum_episode_length=4000, train_start=20, batch_size=128, critic_η=0.001, actor_η=0.001))
# hp = agent("LunarLander-v2", HyperParameter(expl_noise=0.2f0, training_episodes=1000, maximum_episode_length=4000, train_start=20, batch_size=128, critic_η=0.0001, actor_η=0.0001))

# # hp = agent("BipedalWalker-v3", HyperParameter(expl_noise=0.2f0, training_episodes=1000, maximum_episode_length=4000, train_start=20, batch_size=128, critic_η=0.0001, actor_η=0.0001))

function renderEnv(environment, policy, seed=42)

    gym = pyimport("gym")
    
    if environment == "LunarLander-v2"
        global env = gym.make(environment, continuous = true, render_mode="human")
    else
        global env = gym.make(environment, render_mode="human")
    end

    s, info = env.reset(seed=seed)

    R = []
    notSolved = true

    while notSolved
        
        a = action(policy, s, false, EnvParameter(), HyperParameter()) 

        s´, r, terminated, truncated, _ = env.step(a)

        terminated | truncated ? t = true : t = false

        append!(R, r)

        sleep(0.05)
        s = s´
        notSolved = !t
    end
    
    println("The agent has achieved return $(sum(R))")

    env.close()

end #renderEnv

end # module TDDD
