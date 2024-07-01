function y=respond(x,q_b,c_q,v_b,c_v)
    y(1,1)=q_b(c_q,x);
    y(1,2)=v_b(c_v,x);
end