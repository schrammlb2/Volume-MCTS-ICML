import torch

def usher_loss(critic, critic_target, actor, actor_target, preprocced_tuple, transitions,  get_input_tensor, env_params, args):
    inputs_next_norm_tensor, inputs_norm_tensor_pol, inputs_next_norm_tensor_pol = preprocced_tuple
    with torch.no_grad():        
        actions_tensor = torch.tensor(transitions['actions'], dtype=torch.float32)
        r_tensor = torch.tensor(transitions['r'], dtype=torch.float32) 
        exact_goal_tensor = torch.tensor(transitions['exact_goal'], dtype=torch.float32) 
        t = torch.tensor(transitions['t_remaining'], dtype=torch.float32) 
        her_used = torch.tensor(transitions['her_used'], dtype=torch.float32) 
        map_t = lambda t: -1 + 2*t/env_params['max_timesteps']
        # do the normalization
        # concatenate the stuffs
        actions_next = actor(inputs_next_norm_tensor, deterministic=True)
        shape = actions_next.shape
        actions_next = torch.clamp(actions_next, min=-1, max=1)
        clip_return = 1 / (1 - args.gamma)

        q_next_value, p_next_value = critic_target(inputs_next_norm_tensor_pol, map_t(t-1), actions_next, return_p=True)
        q_next_value = q_next_value.detach()
        if args.non_terminal_goals: 
            target_q_value = r_tensor + args.gamma * q_next_value 
        else: 
            target_q_value = r_tensor + args.gamma * q_next_value * (-r_tensor)
        target_q_value = target_q_value.detach()
        target_p_value = p_next_value.detach()

        # target_q_value = torch.clamp(target_q_value, -clip_return, 0)
        

    q0, p0 = critic(inputs_norm_tensor_pol, map_t(t), actions_tensor, return_p=True)

    if args.apply_ratio: 
        c = .01
        p_num = p0.detach()
        p_denom =  p_next_value.detach()

        _, fuzz_input = get_input_tensor(transitions['obs'], transitions['alt_g'], transitions['policy_g'])
        _, fuzz_input_next = get_input_tensor(transitions['obs'], transitions['alt_g'], transitions['policy_g'])

        q_fuzz, p_fuzz = critic(fuzz_input, map_t(t), actions_tensor, return_p=True)
        q_fuzz_next, p_fuzz_next = critic_target(fuzz_input_next, map_t(t), actions_tensor, return_p=True)

        true_c = args.ratio_offset
        q_alpha = .0
        p_alpha = .5
        c = true_c
        def indep_w(alpha, clip=False):  
            x = (p_fuzz.detach() + c)/(alpha*p_fuzz.detach() + (1-alpha)*p_fuzz_next.detach() + c)*her_used + (1-her_used)
            if clip: 
                return alpha*torch.clamp(x, 1/clip_scale, clip_scale)
            else: 
                return alpha*x

        def her_w(alpha, clip=False):  
            x = (p0.detach() + c)/(alpha*p0.detach() + (1-alpha)*p_next_value.detach() + c)*her_used + (1-her_used)
            if clip: 
                return (1-alpha)*torch.clamp(x, 1/clip_scale, clip_scale)
            else: 
                return (1-alpha)*x

        clip_scale = 1+args.ratio_clip
        clip = lambda x: torch.clamp(x, 1/clip_scale, clip_scale)

        _, on_policy_input = get_input_tensor(transitions['obs'], transitions['ag_next'], transitions['policy_g'])
        _ , realized_p = critic(on_policy_input, map_t(t), actions_tensor, return_p=True)
        target_q_fuzz_value = torch.tensor(transitions['alt_r'], dtype=torch.float32)  + args.gamma * q_fuzz_next
        q_loss  = (her_w(q_alpha, clip=True)*(target_q_value - q0).pow(2)).mean()
        q_loss += (indep_w(q_alpha, clip=True)*(target_q_fuzz_value - q_fuzz).pow(2)).mean()
        p_loss = (her_w(p_alpha)*(((p0).pow(2)  - (t-1)/t*(p0*target_p_value))*her_used)).mean()
        p_loss += (indep_w(p_alpha)*her_used*((p_fuzz).pow(2)- (t-1)/t*p_fuzz*p_fuzz_next)).mean() - 2*(realized_p/t).mean()/100

        critic_loss = q_loss + p_loss*clip_return
        # critic_loss = critic_loss + 

    else: 
        true_ratio = 1
        critic_loss = (true_ratio*((target_q_value - q0).pow(2))).mean() + (target_p_value - p0).pow(2).mean()*clip_return
                                                        #p has smaller range, so increase scaling to compensate
    return critic_loss

