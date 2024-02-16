function R = quat2rot(Q)

    n = Q(0+1);
    ex = Q(1+1);
    ey = Q(2+1);
    ez = Q(3+1);
    
    R = eye(3);

    R(0+1, 0+1) = 2 * (n * n + ex * ex) - 1;
    R(0+1, 1+1) = 2 * (ex * ey - n * ez);
    R(0+1, 2+1) = 2 * (ex * ez + n * ey);

    R(1+1, 0+1) = 2 * (ex * ey + n * ez);
    R(1+1, 1+1) = 2 * (n * n + ey * ey) - 1;
    R(1+1, 2+1) = 2 * (ey * ez - n * ex);

    R(2+1, 0+1) = 2 * (ex * ez - n * ey);
    R(2+1, 1+1) = 2 * (ey * ez + n * ex);
    R(2+1, 2+1) = 2 * (n * n + ez * ez) - 1;

end