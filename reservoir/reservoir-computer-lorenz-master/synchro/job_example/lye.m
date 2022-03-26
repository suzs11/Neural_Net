x=rand(1:1:1);u=4;K=-6:0.01:8;y=0;N=10000;
for j=1:N
    x=u.*(x-x.^2);
    df=log(abs((1.0-K)*u*(1-2*x)));
    y=y+df;
end
%Y = y/N;
%save lye.mat K Y -ascii;
set(0,'DefaultFigureVisible','off')
plot(K,y/N,'-r','linewidth',1,'markersize',30);
axis([-6 8 -4 3])
xlabel('\it{\sigma}','FontName','Times New Roman','FontSize',30);
ylabel('\it{\Lambda}','FontName','Times New Roman','FontSize',30);
line([-6,8],[0,0],'linestyle','-','color','k')
set(gca,'FontSize',20);
saveas(gcf,"lye.eps")
close all
