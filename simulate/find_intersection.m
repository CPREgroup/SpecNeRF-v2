% 定义两条曲线
f = @(x) x.^2 - 4;
g = @(x) -2*x + 3;

% 找到两条曲线的交点
x0 = [0 0]; % 用于初始值的矢量
[x, ~, flag] = fsolve(@(x) [f(x(1))-g(x(1)); f(x(2))-g(x(2))], x0);

% 显示交点坐标
if flag > 0
    fprintf('交点坐标为 (%f, %f)\n', x(1), f(x(1)));
    fprintf('交点坐标为 (%f, %f)\n', x(2), f(x(2)));
else
    fprintf('未找到交点\n');
end
