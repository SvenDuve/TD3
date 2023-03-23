import TD3


# hp = TD3.agent("Pendulum-v1", TD3.HyperParameter(expl_noise=0.1f0, noise_clip=0.5f0, training_episodes=400, maximum_episode_length=3000, train_start=20, batch_size=128))
#hp = TD3.agent("Pendulum-v1", TD3.HyperParameter(expl_noise=0.1f0, noise_clip=0.5f0, training_episodes=400, maximum_episode_length=3000, train_start=20, batch_size=128))


results = []



for i in collect(1:10)

    hp = TD3.agent("LunarLander-v2", TD3.HyperParameter(expl_noise=0.2f0, training_episodes=1000, maximum_episode_length=3000, train_start=20, batch_size=256, critic_η=0.001, actor_η=0.0001))
    push!(results, hp)

end


series = hcat([results[i].episode_reward for i in 1:10]...)


using Plots
using Statistics

x = 1:size(series)[1]



y = [mean(series[i,1:10]) for i in x]
y_plus = [mean(series[i,1:10]) for i in x] .+ [std(series[i,1:10]) for i in x]
y_minus = [mean(series[i,1:10]) for i in x] .- [std(series[i,1:10]) for i in x]


plot(x, y, label = "mean", title="TD3")
plot!(x, y_plus, label = "upper std")
plot!(x, y_minus, label = "lower std")


using DelimitedFiles


# save matrix to a CSV file
# writedlm("pendulum_TD3.csv", series, ',')



# save matrix to a CSV file
writedlm("lunarlander_DDPG.csv", hp.episode_reward, ',')

# load matrix from the CSV file
# B = readdlm("pendulum_DDPG.csv", ',')

using BSON


using BSON: @save

for (episode, model) in zip(collect(100:100:1000), hp.trained_agents)
    @save "LL_" * string(episode) * ".bson" model
end


using BSON: @load
using Flux

@load "LL_1000.bson" model