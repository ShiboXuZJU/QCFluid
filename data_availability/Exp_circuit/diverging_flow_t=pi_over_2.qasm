OPENQASM 2.0;
include "qelib1.inc";
qreg q[10];
u(1.4014304319964814,-1.2411064982374729,-0.2227860887779496) q[0];
u(1.4637152621887246,3.074905887196458,3.0832822394749044) q[4];
barrier q[0],q[4],q[1],q[2],q[3],q[5],q[9],q[6],q[7],q[8];
cz q[4],q[0];
barrier q[0],q[4],q[1],q[2],q[3],q[5],q[9],q[6],q[7],q[8];
u(1.8886275797963898,-0.5463525020924465,2.191947801702619) q[0];
u(1.5707967281341557,-1.0556180040028433,-9.417494362118362e-07) q[0];
u(1.0005976198106108,-1.2499351517197508,0.5132752776342393) q[1];
u(1.7944532929972263,-2.2023098724802836,2.5585163869455343) q[4];
barrier q[0],q[4],q[1],q[2],q[3],q[5],q[9],q[6],q[7],q[8];
cz q[4],q[1];
barrier q[0],q[4],q[1],q[2],q[3],q[5],q[9],q[6],q[7],q[8];
u(0.012090825700303273,1.9500067538130548,-1.1096447920373471) q[1];
u(1.5703778662153127,1.972823883608747,0.8187777418377347) q[2];
u(0.3185255619761624,-0.40976010251055506,2.832160691560274) q[3];
u(1.5312006258900357,-1.6882350323622326,-0.002059635288095496) q[4];
barrier q[0],q[4],q[1],q[2],q[3],q[5],q[9],q[6],q[7],q[8];
cz q[3],q[1];
cz q[2],q[4];
barrier q[0],q[4],q[1],q[2],q[3],q[5],q[9],q[6],q[7],q[8];
u(1.334802044148674,-1.5734488159513114,0.41294346335066656) q[1];
u(1.5789362392649668,-2.647337330836031,1.2343382757004262) q[2];
u(0.1496843149161416,0.25240076925486354,-2.732588442730405) q[3];
u(3.104081781599586,0.14940730559112936,-2.90062226255373) q[4];
barrier q[0],q[4],q[1],q[2],q[3],q[5],q[9],q[6],q[7],q[8];
cz q[4],q[1];
cz q[2],q[3];
barrier q[0],q[4],q[1],q[2],q[3],q[5],q[9],q[6],q[7],q[8];
u(0.5573333001789577,1.572316927912218,1.57003135992945) q[1];
u(0.2957504859539476,-0.5660081957466958,0.9529996611206304) q[2];
u(1.062490642772202,2.143673890242571,-0.23640405326495006) q[3];
u(1.7653177413851524,-1.6970817388062578,2.9920379767220995) q[4];
barrier q[0],q[4],q[1],q[2],q[3],q[5],q[9],q[6],q[7],q[8];
cz q[3],q[1];
cz q[2],q[4];
barrier q[0],q[4],q[1],q[2],q[3],q[5],q[9],q[6],q[7],q[8];
u(0.1478979587554933,pi/2,-pi/2) q[1];
u(0.06084838922022684,-0.03706161222903548,0.6032475567051154) q[2];
u(9.541398640621977e-08,-0.15085042455843878,0.150850799484036) q[2];
u(0.012424147906859922,-0.2522033244199755,-1.1923060107492347) q[4];
u(0.6115468974794065,-2.5488504916842754,-1.1965014618480028) q[4];
u(1.5707963705062866,3.141592566167013,-3.1415925661670134) q[5];
u(1.5707967281341557,-1.0556180040028433,-9.417494362118362e-07) q[5];
u(1.5707963705062866,0.19634954631328583,-0.19634954631328583) q[9];
u(0.6115468974794065,-2.5488504916842754,-1.1965014618480028) q[9];
barrier q[0],q[4],q[1],q[2],q[3],q[5],q[9],q[6],q[7],q[8];
cz q[4],q[0];
cz q[3],q[1];
cz q[5],q[9];
barrier q[0],q[4],q[1],q[2],q[3],q[5],q[9],q[6],q[7],q[8];
u(2.2302015940987762e-07,-1.4628079428282081,-1.040775370387581) q[0];
u(1.0614144865853816,0.003323296649468066,-1.5771679574941126) q[1];
u(1.570796051616045,0.46120471994731727,3.1415920016016674) q[1];
u(0.5403441888221413,0.03164961882989825,-2.1798438525953223) q[3];
u(6.843186916831291e-07,0.6978668191000192,-0.6978657228123586) q[3];
u(1.570796223023983,0.3825382565497435,0.4890688135532604) q[4];
u(2.2302015940987762e-07,-1.4628079428282081,-1.040775370387581) q[5];
u(1.5707963705062866,1.5707963705062866,-1.5707963705062866) q[6];
u(1.570796051616045,0.46120471994731727,3.1415920016016674) q[6];
u(1.570796223023983,0.3825382565497435,0.4890688135532604) q[9];
barrier q[0],q[4],q[1],q[2],q[3],q[5],q[9],q[6],q[7],q[8];
cz q[4],q[1];
cz q[6],q[9];
barrier q[0],q[4],q[1],q[2],q[3],q[5],q[9],q[6],q[7],q[8];
u(6.109476089477633e-07,2.3757652969182503,0.26388157813838387) q[1];
u(1.570793601092359,-2.2021549015164816,2.759055253421076) q[4];
u(6.109476089477633e-07,2.3757652969182503,0.26388157813838387) q[6];
u(1.570793601092359,-2.2021549015164816,2.759055253421076) q[9];
barrier q[0],q[4],q[1],q[2],q[3],q[5],q[9],q[6],q[7],q[8];
cz q[4],q[0];
cz q[5],q[9];
barrier q[0],q[4],q[1],q[2],q[3],q[5],q[9],q[6],q[7],q[8];
u(1.5707964332354443,3.141590889509864,1.203006394828419) q[0];
u(1.570796239104274,0.9654227684475103,2.202101818778485) q[4];
u(1.5707964332354443,3.141590889509864,1.203006394828419) q[5];
u(1.570796239104274,0.9654227684475103,2.202101818778485) q[9];
barrier q[0],q[4],q[1],q[2],q[3],q[5],q[9],q[6],q[7],q[8];
cz q[4],q[1];
cz q[6],q[9];
barrier q[0],q[4],q[1],q[2],q[3],q[5],q[9],q[6],q[7],q[8];
u(1.570812523529767,-2.9785720423447515e-07,-3.100852329768564) q[1];
u(1.8438652697604332,-3.105034111679521,1.621851318899214) q[4];
u(1.570812523529767,-2.9785720423447515e-07,-3.100852329768564) q[6];
u(1.5707963705062866,0.7853981852531433,-0.7853981852531433) q[7];
u(9.541398640621977e-08,-0.15085042455843878,0.150850799484036) q[7];
u(1.5707963705062866,0.39269909262657166,-0.39269909262657166) q[8];
u(6.843186916831291e-07,0.6978668191000192,-0.6978657228123586) q[8];
u(1.8438652697604332,-3.105034111679521,1.621851318899214) q[9];
barrier q[0],q[4],q[1],q[2],q[3],q[5],q[9],q[6],q[7],q[8];
