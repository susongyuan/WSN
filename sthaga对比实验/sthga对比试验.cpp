#define _CRT_SECURE_NO_WARNINGS

/*
代码来自论文 "采用前向编码方案的混合遗传算法，在无线传感器网络中实现寿命最大化，"
发表在IEEE进化计算期刊，2010年。

提醒：
1. 在readInst()中更改输入文件名
2. 更改节点数量，例如 #define NODEN 100
3. 此代码用于 '区域覆盖'
*/

/*
【代码说明】

本代码采用混合遗传算法（STHGA）来优化无线传感器网络的覆盖，以最大化传感器网络的寿命。
传感器能量被拆分成多个子传感器（每个能量为1），当一个物理传感器所有子传感器的能量耗尽时，
该物理传感器将失效，从而影响其所属的覆盖集。
代码中输出的统计指标包括：
  1. 最佳完整覆盖集数量（Means）；
  2. 平均每个子集使用的传感器数（Average Sensors per Subset）；
  3. 试验的总耗时（Total Trial Time）和整个案例的总体耗时（Overall Case Time）。

【最新修改】
- 将 `init_grid()` 改为与原代码类似的方式，动态计算网格数量与步长(stepL, stepW)，
  而不是固定 50×50 的网格。
- 一旦满足 `pbest.csetN == apmin`，就提前退出，不再强制跑满 MAX_EVA。
*/

#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>
#include <direct.h>
#include <stdbool.h>

// ============ 宏定义区域 ============
#define NODEN      200000      // 最大传感器节点数，已增加以支持能量拆分
#define MAX_Parts  20          // 能量最大值
#define TRIALNUM   30          // 测试次数
#define FMAX       100000      // 原始常量
#define UPAPMIN    10000       // 覆盖最小场地的次数的最大值
#define CRIMAX     10000       // 覆盖最少场地的最大数量
#define GRID_C     600          // 如果不再使用，这里可留作兼容
#define GRID_D     600          // 如果不再使用，这里可留作兼容
#define PI         3.1415926535897932384626
#define POPSIZE    30           // 遗传算法中个体数量，即每一代的种群规模

// 新增或保留：网格分割级别
#define GRID_LEVEL 4           // 将传感器半径 R 分为 GRID_LEVEL 等分来离散化

const int CMAX = GRID_LEVEL* GRID_LEVEL * 4;  // 原代码里用于存储覆盖网格索引的数组大小

// 全局变量
double R;        // 动态设置的半径
double L = 50.0; // 区域长度（可根据案例设为5或10或其他）
double W = 50.0; // 区域宽度（可根据案例设为5或10或其他）

int tmp_arr[UPAPMIN];
int nos[UPAPMIN][UPAPMIN];
int N = 0; // 初始化为0，后续设置为实际传感器数量
int ino;

int OPos[NODEN][CMAX + 1][2];
int GRID_L;      // 动态计算得到的网格数量( x 方向 )
int GRID_W;      // 动态计算得到的网格数量( y 方向 )

int trialNum;
double stepL;    // 网格在 x 方向的大小
double stepW;    // 网格在 y 方向的大小

double point[NODEN][2];
int sensorEnergy[NODEN];
double yita;
int parts;

int fieldNum;
int OPosf[NODEN][CMAX + 1];
int field[FMAX];
int grid[GRID_C][GRID_D];       // 仅保留原格式（若需要也可改用动态分配）
int fieldIndex[GRID_C][GRID_D]; // 仅保留原格式

int G;
int apmin;
int countEvals;
clock_t tp1, tp2;

char filename[100], filename2[100], filename3[100], filename4[100], filename5[100];
char filename6[30];
FILE *ftotal, *finst, *favgtotal;
FILE *fcrate;
int evalStep = 10;
int recordCount;
int maxRecordCount;
double evalValueAvg[100000];

// 遗传算法相关参数
int K1 = 5;
int K2 = 5;
int Gm = 20;
double MUI = 0.5;

// 新增：MAX_EVA 用于设置当前案例的评估上限（不同案例使用不同值）
int MAX_EVA = 0;

// ============ 结构体定义 ============
struct Population {
    int d[NODEN][MAX_Parts]; // 调度序列，支持子传感器
    int csetN;               // 完整覆盖集的数量
    double fitness;          // 适应度 = csetN + 不完整集的覆盖百分比（即 csetN+1 集）
} P[POPSIZE * 2], pbest;

int NP2 = POPSIZE * 2;

struct node {
    int no;
    int part;
    struct node* next;
};

int cricount;
struct node criti[CRIMAX];

// 案例结构体
struct Case {
    int No;
    double R;
    int N;
};

//-------------------------------------随机数生成函数-------------------------------------
double rand01() {
    return (double)rand() / (double)(RAND_MAX + 1);
}

//-------------------------------------判断是否覆盖-------------------------------------
// 可按你自己的需求判断覆盖，比如检查传感器圆是否包含网格单元。
// 这里示例仅检查“网格左下角 + 右下角 + 左上角 + 右上角”这4点都在圆内，才算覆盖。
int iscovered(int i, int j, int k) {
    double x     = i * stepL;
    double y     = j * stepW;
    double x2    = x + stepL;
    double y2    = y + stepW;
    double cx    = point[k][0];
    double cy    = point[k][1];
    double r2    = R * R;

    // 判断4个顶点是否在圆内
    if ( (x - cx)*(x - cx) + (y - cy)*(y - cy)   > r2 ) return 0;
    if ( (x2 - cx)*(x2 - cx) + (y - cy)*(y - cy) > r2 ) return 0;
    if ( (x - cx)*(x - cx) + (y2 - cy)*(y2 - cy) > r2 ) return 0;
    if ( (x2 - cx)*(x2 - cx) + (y2 - cy)*(y2 - cy) > r2 ) return 0;

    return 1;
}

//-------------------------------------初始化网格覆盖 (动态分割)-------------------------------------
// 将原先固定 50×50，step=1.0 的做法，改为根据 R/GRID_LEVEL 动态划分
void init_grid(double current_R) {
    // 1) 依据传感器半径 R 和 GRID_LEVEL，计算分辨率 step
    double step = current_R / GRID_LEVEL;

    // 2) 动态计算网格数量 GRID_L, GRID_W
    //    如果 L=5/W=5 或 L=10/W=10，会得到不同的 (GRID_L, GRID_W)
    GRID_L = (int)(L / step);
    GRID_W = (int)(W / step);

    if (GRID_L <= 0) GRID_L = 1;
    if (GRID_W <= 0) GRID_W = 1;

    // 3) 计算网格大小 stepL, stepW
    stepL = L / GRID_L;
    stepW = W / GRID_W;

    // 4) 对每个传感器 k，枚举其周围可能的网格，用 iscovered() 判断是否覆盖
    for (int k = 0; k < N; k++) {
        OPos[k][CMAX][0] = 0;

        // 传感器中心 (cx, cy)
        double cx = point[k][0];
        double cy = point[k][1];

        // 设定网格起点 i0, j0 (略向左下取整)
        int i0 = (int)((cx - current_R) / stepL);
        int j0 = (int)((cy - current_R) / stepW);
        if (i0 < 0) i0 = 0;
        if (j0 < 0) j0 = 0;

        // 枚举 (i, j) 范围：保证覆盖到方圆 current_R 的网格
        // 这里给出 i0 ~ i0 + 2*GRID_LEVEL
        // 也可根据 (cx + R)/stepL 来取上界
        int i_max = i0 + GRID_LEVEL * 2;
        int j_max = j0 + GRID_LEVEL * 2;

        if (i_max > GRID_L) i_max = GRID_L;
        if (j_max > GRID_W) j_max = GRID_W;

        for (int i = i0; i < i_max; i++) {
            for (int j = j0; j < j_max; j++) {
                if (iscovered(i, j, k)) {
                    int index = OPos[k][CMAX][0];  // 当前记录的位置
                    OPos[k][index][0] = i;
                    OPos[k][index][1] = j;
                    OPos[k][CMAX][0] += 1;
                }
            }
        }
    }
}

//-------------------------------------计算字段覆盖-------------------------------------
void calc_field() {
    int i, j, k, h;
    int fh[FMAX];
    int count = 1;
    memset(fh, -1, sizeof(fh));
    fieldNum = 0;

    // grid[][] 数组清零
    // 由于 max(GRID_L, GRID_W) 可能比 50 大，你可视需要动态改写 grid[][], fieldIndex[][] 。
    // 这里先保留 grid_C, grid_D 为固定 50 只是示例；若需更大，请自行改成动态内存。
    // 为避免越界，这里只初始化一个 50×50 的 grid，如果 R=10, L=10, step=2.5 -> GRID_L=4 => 不越界
    // 但如果 R 很大，会导致 i,j 超过 50，需要再行修改。
    for (i = 0; i < 50; i++) {
        for (j = 0; j < 50; j++) {
            grid[i][j] = 0;
        }
    }

    // 将传感器覆盖到的网格打上“临时field”记号
    for (h = 0; h < N; h++) {
        for (k = 0; k < OPos[h][CMAX][0]; k++) {
            i = OPos[h][k][0];
            j = OPos[h][k][1];
            // 要保证 (i < 50 && j < 50) 否则会越界
            if (i < 50 && j < 50 && grid[i][j] == 0) {
                grid[i][j] = count++;
                if (count > FMAX) {
                    printf("Error: FMAX is not big enough\n");
                    exit(1);
                }
            }
        }
    }

    printf("Total fields: %d\n", fieldNum);

    for (i = 0; i < 50; i++) {
        for (j = 0; j < 50; j++) {
            if (grid[i][j] != 0 && fh[grid[i][j]] == -1) {
                fh[grid[i][j]] = fieldNum++;
            }
            fieldIndex[i][j] = fh[grid[i][j]];
        }
    }

    printf("Total fields: %d\n", fieldNum);

    // OPosf[h][CMAX] = 0 表示 sensor h 覆盖的所有 field ID
    for (h = 0; h < N; h++) {
        OPosf[h][CMAX] = 0;
        for (k = 0; k < OPos[h][CMAX][0]; k++) {
            i = OPos[h][k][0];
            j = OPos[h][k][1];
            if (i >= 50 || j >= 50) {
                // 超出固定 grid 范围就略过
                continue;
            }
            int fieldID = fieldIndex[i][j];
            int alreadyAdded = 0;
            for (int m = 0; m < OPosf[h][CMAX]; m++) {
                if (OPosf[h][m] == fieldID) {
                    alreadyAdded = 1;
                    break;
                }
            }
            if (!alreadyAdded && fieldID >= 0) {
                OPosf[h][ OPosf[h][CMAX]++ ] = fieldID;
            }
        }
    }

    // 简单检查每个 field 的被覆盖次数
    int coverage[FMAX] = { 0 };
    for (h = 0; h < N; h++) {
        for (k = 0; k < OPosf[h][CMAX]; k++) {
            coverage[ OPosf[h][k] ]++;
        }
    }
    printf("Field coverage statistics:\n");
    for (i = 1; i <= fieldNum; i++) {
        printf("Field %d is covered by %d sensor(s)\n", i, coverage[i]);
    }
}

//-------------------------------------评估覆盖-------------------------------------
double eval_cover(int scheIndex, int id) {
    int i, j, k;
    int count = 0;

    for (k = 0; k < N; k++) {
        parts = sensorEnergy[k];
        for (int p = 0; p < parts; p++) {
            if (P[id].d[k][p] == scheIndex) {
                // 将 sensor k 覆盖的所有 field 置为 1
                for (j = 0; j < OPosf[k][CMAX]; j++) {
                    field[ OPosf[k][j] ] = 1;
                }
            }
        }
    }

    for (i = 0; i < fieldNum; i++) {
        if (field[i] == 1) {
            count++;
            field[i] = 0; // 复位
        }
    }
    return (double)count / (double)(fieldNum);
}

//-------------------------------------混合调度-------------------------------------
void mixSche(int id) {
    int i, j, k, tmp_val, part;
    double v;

    for (i = 0; i < K2; i++) {
        tmp_val = (int)(rand01() * N);
        part = (int)(rand01() * MAX_Parts);
        if (tmp_val >= N || part >= MAX_Parts) continue;
        k = P[id].d[tmp_val][part];

        if (k == -1) continue;

        if (k != P[id].csetN) {
            P[id].d[tmp_val][part] = -1;
            if (eval_cover(k, id) == 1.0) {
                j = (int)(rand01() * P[id].csetN);
                if (j == k) j = P[id].csetN;
                P[id].d[tmp_val][part] = j;
            } else {
                P[id].d[tmp_val][part] = k;
            }
        } else {
            v = eval_cover(k, id);
            P[id].d[tmp_val][part] = (int)(rand01() * P[id].csetN);
            if (eval_cover(k, id) < v) {
                P[id].d[tmp_val][part] = P[id].csetN;
            }
        }
    }
}

//-------------------------------------前向调度-------------------------------------
void forwSche(int id) {
    int i, k, part, tmp_val;
    for (i = 0; i < K1; i++) {
        tmp_val = (int)(rand01() * N);
        part    = (int)(rand01() * MAX_Parts);
        if (tmp_val >= N || part >= MAX_Parts) continue;

        k = P[id].d[tmp_val][part];
        while (k == P[id].csetN || k == -1) {
            tmp_val = (int)(rand01() * N);
            part    = (int)(rand01() * MAX_Parts);
            if (tmp_val >= N || part >= MAX_Parts) break;
            k = P[id].d[tmp_val][part];
        }
        if (tmp_val >= N || part >= MAX_Parts) continue;

        P[id].d[tmp_val][part] = P[id].csetN;
        if (eval_cover(k, id) != 1.0) {
            P[id].d[tmp_val][part] = k;
        }
    }
}

//-------------------------------------关键调度-------------------------------------
void critSche(int id) {
    int i, j, k, selectedPart;
    int sensorIndex;
    bool foundRedundant;
    struct node* pt;

    for (i = 0; i < cricount; i++) {
        for (j = 0; j <= apmin; j++) {
            tmp_arr[j] = 0;
        }
        pt = &criti[i];
        while (pt->next != NULL) {
            pt = pt->next;
            int currentSet = P[id].d[ pt->no ][0];
            nos[currentSet][ tmp_arr[currentSet] ] = pt->no;
            tmp_arr[currentSet]++;
        }

        if (tmp_arr[P[id].csetN] == 0) {
            j = 0;
            foundRedundant = false;
            while (j < P[id].csetN && !foundRedundant) {
                if (tmp_arr[j] >= 2) {
                    for (int index = 0; index < tmp_arr[j]; index++) {
                        sensorIndex = nos[j][index];
                        for (selectedPart = 0; selectedPart < sensorEnergy[sensorIndex]; selectedPart++) {
                            if (P[id].d[sensorIndex][selectedPart] == j) {
                                P[id].d[sensorIndex][selectedPart] = -1;
                                if (eval_cover(j, id) == 1.0) {
                                    P[id].d[sensorIndex][selectedPart] = P[id].csetN;
                                    foundRedundant = true;
                                    break;
                                } else {
                                    P[id].d[sensorIndex][selectedPart] = j;
                                }
                            }
                        }
                        if (foundRedundant) {
                            break;
                        }
                    }
                }
                j++;
            }
        }
    }
}

//-------------------------------------遗传算法优化-------------------------------------
void GAoptimize() {
    int i, j, k, part, a1, a2;
    double v;
    double bv;

    // (1) 交叉与选择
    for (i = POPSIZE; i < NP2; i++) {
        a1 = (int)(rand01() * POPSIZE);
        a2 = (int)(rand01() * POPSIZE);
        while (a2 == a1) a2 = (int)(rand01() * POPSIZE);

        for (j = 0; j < N; j++) {
            parts = sensorEnergy[j];
            for (part = 0; part < parts; part++) {
                if (rand01() < 0.5)
                    P[i].d[j][part] = P[a1].d[j][part];
                else
                    P[i].d[j][part] = P[a2].d[j][part];
            }
        }

        // 评估新个体
        for (k = 0; k <= apmin; k++) {
            v = eval_cover(k, i);
            if (v != 1.0) {
                P[i].csetN = k;
                P[i].fitness = k + v;
                break;
            }
        }

        countEvals++;
        if (countEvals % evalStep == 1) {
            evalValueAvg[recordCount] += pbest.fitness;
            recordCount++;
        }

        // 竞争选择
        if (P[a1].fitness < P[a2].fitness) a1 = a2;
        if (P[i].fitness < P[a1].fitness) {
            P[i].csetN = P[a1].csetN;
            for (j = 0; j < N; j++) {
                parts = sensorEnergy[j];
                for (part = 0; part < parts; part++) {
                    P[i].d[j][part] = P[a1].d[j][part];
                }
            }
            P[i].fitness = P[a1].fitness;
        } else if (pbest.fitness < P[i].fitness) {
            pbest.fitness = P[i].fitness;
            pbest.csetN = P[i].csetN;
            for (j = 0; j < N; j++) {
                parts = sensorEnergy[j];
                for (part = 0; part < parts; part++) {
                    pbest.d[j][part] = P[i].d[j][part];
                }
            }
        }
    }

    // (2) 变异
    if (G % Gm == 0) {
        bv = 0;
        for (i = POPSIZE; i < NP2; i++) {
            if (P[i].fitness > bv) bv = P[i].fitness;
        }
        if (bv > ((int)bv)) {
            for (i = POPSIZE; i < NP2; i++) {
                if (P[i].fitness == bv) {
                    for (j = 0; j < N; j++) {
                        parts = sensorEnergy[j];
                        for (k = 0; k < parts; k++) {
                            if (P[i].d[j][k] == P[i].csetN && rand01() < MUI) {
                                P[i].d[j][k] = (int)(rand01() * (P[i].csetN));
                            }
                        }
                    }
                    v = eval_cover(P[i].csetN, i);
                    P[i].fitness = P[i].csetN + v;
                    countEvals++;
                    if (countEvals % evalStep == 1) {
                        evalValueAvg[recordCount] += pbest.fitness;
                        recordCount++;
                    }
                }
            }
        }
    }

    // (3) 混合调度/前向调度/关键调度
    for (i = POPSIZE; i < NP2; i++) mixSche(i);
    for (i = POPSIZE; i < NP2; i++) forwSche(i);
    for (i = POPSIZE; i < NP2; i++) critSche(i);

    // (4) 最终评估
    for (i = POPSIZE; i < NP2; i++) {
        v = eval_cover(P[i].csetN, i);
        P[i].fitness = P[i].csetN + v;
        if (v == 1.0) P[i].csetN++;
        countEvals++;
        if (countEvals % evalStep == 1) {
            evalValueAvg[recordCount] += pbest.fitness;
            recordCount++;
        }
        if (P[i].fitness > pbest.fitness) {
            pbest.fitness = P[i].fitness;
            pbest.csetN = P[i].csetN;
            for (j = 0; j < N; j++) {
                parts = sensorEnergy[j];
                for (k = 0; k < parts; k++) {
                    pbest.d[j][k] = P[i].d[j][k];
                }
            }
        }
    }

    // (5) 新群体替换旧群体
    for (i = 0; i < POPSIZE; i++) {
        P[i].csetN = P[POPSIZE + i].csetN;
        P[i].fitness = P[POPSIZE + i].fitness;
        for (j = 0; j < N; j++) {
            parts = sensorEnergy[j];
            for (k = 0; k < parts; k++) {
                P[i].d[j][k] = P[POPSIZE + i].d[j][k];
            }
        }
    }
}

//-------------------------------------初始化种群-------------------------------------
void init_population(int apmin_current) {
    int i, k, part;
    double v;
    for (k = 0; k < POPSIZE; k++) {
        for (i = 0; i < N; i++) {
            parts = sensorEnergy[i];
            for (part = 0; part < parts; part++) {
                P[k].d[i][part] = rand() % apmin_current;
            }
        }
        P[k].csetN = apmin_current;
        P[k].fitness = 0.0;
    }

    pbest.csetN = 0;
    pbest.fitness = 0.0;

    for (i = 0; i < POPSIZE; i++) forwSche(i);

    double v_cover = eval_cover(1, 0);
    if (v_cover == 1.0) {
        P[0].csetN++;
    }
    P[0].fitness = P[0].csetN + v_cover;

    pbest.csetN = P[0].csetN;
    pbest.fitness = P[0].fitness;
    for (i = 0; i < N; i++) {
        parts = sensorEnergy[i];
        for (part = 0; part < parts; part++) {
            pbest.d[i][part] = P[0].d[i][part];
        }
    }
    countEvals++;
    if (countEvals % evalStep == 1) {
        evalValueAvg[recordCount] += pbest.fitness;
        recordCount++;
    }

    for (i = 1; i < POPSIZE; i++) {
        double v_cover_i = eval_cover(1, i);
        if (v_cover_i == 1.0) {
            P[i].csetN++;
        }
        P[i].fitness = P[i].csetN + v_cover_i;

        countEvals++;
        if (countEvals % evalStep == 1) {
            evalValueAvg[recordCount] += pbest.fitness;
            recordCount++;
        }

        if (P[i].fitness > pbest.fitness) {
            pbest.fitness = P[i].fitness;
            pbest.csetN = P[i].csetN;
            for (k = 0; k < N; k++) {
                parts = sensorEnergy[k];
                for (part = 0; part < parts; part++) {
                    pbest.d[k][part] = P[i].d[k][part];
                }
            }
        }
    }
}

//-------------------------------------计算关键传感器-------------------------------------
void calc_criti() {
    int i, k, h, part;
    int cc;
    struct node* pt;

    cricount = 0;
    for (i = 0; i < fieldNum; i++) {
        if (field[i] == apmin) {
            cc = 0;
            pt = &criti[cricount];
            for (k = 0; k < N; k++) {
                parts = sensorEnergy[k];
                for (part = 0; part < MAX_Parts; part++) {
                    if (P[0].d[k][part] == -1) continue;
                    for (h = 0; h < OPosf[k][CMAX]; h++) {
                        if (OPosf[k][h] == i) {
                            pt->next = (struct node*)malloc(sizeof(struct node));
                            if (pt->next == NULL) {
                                printf("Memory allocation failed!\n");
                                exit(1);
                            }
                            pt = pt->next;
                            pt->no   = k;
                            pt->part = part;
                            pt->next = NULL;
                            cc++;
                            break;
                        }
                    }
                    if (cc == apmin) break;
                }
                if (cc == apmin) break;
            }
            cricount++;
            if (cricount >= CRIMAX) {
                printf("CRIMAX is not big enough!\n");
                exit(0);
            }
        }
    }
}

//-------------------------------------检查部署-------------------------------------
bool deployOK(int apmin_required) {
    int i, j, k;
    bool flag = true;

    for (i = 0; i < fieldNum; i++) {
        field[i] = 0;
    }

    for (k = 0; k < N; k++) {
        for (j = 0; j < OPosf[k][CMAX]; j++) {
            int fieldID = OPosf[k][j];
            field[fieldID]++;
        }
    }

    apmin = field[0];
    for (i = 0; i < fieldNum; i++) {
        if (field[i] > 0) {
            if (apmin > field[i]) {
                apmin = field[i];
            }
        } else {
            flag = false;
            break;
        }
    }

    // 将 apmin 至少设置为当前案例的要求
    if (apmin < apmin_required) {
        flag = false;
    }

    if (apmin > UPAPMIN) {
        printf("UPAPMIN is not big enough!\n");
        exit(1);
    }

    if (flag) {
        calc_criti();
    }

    for (i = 0; i < fieldNum; i++) {
        field[i] = 0;
    }
    return flag;
}

//-------------------------------------读取并拆分传感器-------------------------------------
void readInst(const char* input_filename) {
    strcpy(filename, input_filename);
    FILE* fin = fopen(filename, "r");
    if (fin == NULL) {
        printf("Error: Unable to open file %s\n", filename);
        exit(1);
    }

    double temp_point[NODEN][2];
    int temp_energy[NODEN];
    int original_N = 0;

    while (fscanf(fin, "%lf %lf %d", &temp_point[original_N][0],
                                    &temp_point[original_N][1],
                                    &temp_energy[original_N]) == 3) {
        original_N++;
        if (original_N >= NODEN) {
            printf("Error: Number of sensors exceeds NODEN=%d\n", NODEN);
            exit(1);
        }
    }
    fclose(fin);

    printf("Original number of sensors: %d\n", original_N);

    // 能量拆分
    int new_N = 0;
    for (int i = 0; i < original_N; i++) {
        int Ei = temp_energy[i];
        int sub_sensors = (int)ceil((double)Ei);
        for (int j = 0; j < sub_sensors; j++) {
            if (new_N >= NODEN) {
                printf("Error: Number of sub-sensors exceeds NODEN=%d\n", NODEN);
                exit(1);
            }
            point[new_N][0] = temp_point[i][0];
            point[new_N][1] = temp_point[i][1];
            sensorEnergy[new_N] = 1; // 每个子传感器能量=1
            // 初始化染色体
            for (int part = 0; part < sensorEnergy[new_N]; part++) {
                P[0].d[new_N][part] = -1;
            }
            new_N++;
        }
    }

    N = new_N;
    printf("Total number of sub-sensors after energy splitting: %d\n", N);
}

//-------------------------------------主函数-------------------------------------
int main() {
    struct Case cases[] = {
        {1,  5.0,  500},
        {2, 10.0,  500},
        {3,  5.0, 1000},
        {4, 10.0, 1000},
        {5,  5.0, 2500},
        {6, 10.0, 2500},
        {7,  5.0, 5000},
        {8, 10.0, 5000},
        {9,  5.0,10000},
        {10,10.0,10000},
    };
    int num_cases = sizeof(cases) / sizeof(cases[0]);

    FILE* fsummary = fopen("summary.txt", "w");
    if (fsummary == NULL) {
        printf("Error: Unable to create summary.txt\n");
        exit(1);
    }

    fprintf(fsummary, "Cases\tsthga\n");
    fprintf(fsummary, "No.\tR\tN\tMeans\taverage\tTotalTime(ms)\n");

    srand((unsigned)time(0));

    for (int c = 0; c < num_cases; c++) {
        struct Case current_case = cases[c];
        printf("Processing Case %d: R=%.1f, N=%d\n", current_case.No, current_case.R, current_case.N);

        // 动态设置 R
        R = current_case.R;

        // 设置输入文件名
        char input_filename[50];
        sprintf(input_filename, "input_%d.txt", current_case.N);

        clock_t total_start_case = clock();

        // 读取并拆分
        readInst(input_filename);

        // 这里使用动态的 init_grid(R)
        init_grid(R);

        // 计算字段
        calc_field();

        // 至少覆盖2次
        bool deployment_success = deployOK(2);
        printf("Deploying : %d\n", deployment_success ? 1 : 0);
        printf("apmin = %d\n", apmin);

        if (deployment_success) {
            switch (current_case.N) {
            case 40:    MAX_EVA = 15000; break;
            case 60:    MAX_EVA = 15000; break;
            case 100:   MAX_EVA = 15000; break;
            case 140:   MAX_EVA = 15000; break;
            case 180:   MAX_EVA = 15000; break;
            case 200:   MAX_EVA = 15000; break;
            case 250:   MAX_EVA = 15000; break;
            case 300:   MAX_EVA = 15000; break;
            case 500:   MAX_EVA = 15000; break;
            case 600:   MAX_EVA = 15000; break;
            case 800:   MAX_EVA = 15000; break;
            case 1000:  MAX_EVA = 15000; break;
            case 1500:  MAX_EVA = 15000; break;
            case 2500:  MAX_EVA = 15000; break;
            case 5000:  MAX_EVA = 15000; break;
            case 10000: MAX_EVA = 30000; break;
            case 15000: MAX_EVA = 45000; break;
            case 20000: MAX_EVA = 60000; break;
            case 25000: MAX_EVA = 75000; break;
            case 30000: MAX_EVA = 90000; break;
            default: printf("MAX_EVA error!\n"); MAX_EVA = -1;
            }
            printf("MAX_EVA = %d\n", MAX_EVA);

            yita = N * PI * R * R / (double)apmin / L / W;

            int countOK    = 0;
            int gBest      = 0;
            int gWorst     = N;
            int countResult= 0;
            int avgEval    = 0;
            maxRecordCount = 0;
            recordCount    = 0;

            long long total_trial_time_ticks = 0;

            strcpy(filename2, "inst.txt");
            strcpy(filename3, "avgEval.txt");
            strcpy(filename6, "crate.txt");

            finst     = fopen(filename2, "a");
            favgtotal = fopen(filename3, "a");
            fcrate    = fopen(filename6, "a");

            if (finst == NULL || favgtotal == NULL || fcrate == NULL) {
                printf("Error: Unable to open one of the output files.\n");
                exit(1);
            }

            for (trialNum = 0; trialNum < TRIALNUM; trialNum++) {
                G          = 0;
                countEvals = 0;
                recordCount= 0;

                init_population(apmin);

                clock_t trial_start = clock();

                // 若找到 pbest.csetN == apmin 即可提前退出
                while (countEvals <= MAX_EVA) {
                    GAoptimize();
                    G++;

                    if (G % 20 == 0) {
                        clock_t current_time = clock();
                        int elapsed_ms = (int)((double)(current_time - trial_start) / CLOCKS_PER_SEC * 1000);
                        printf("Case %d, Trial %d:\tG=%d\tEvals=%d\tcsetN=%d\tTime=%dms\n",
                               current_case.No, trialNum + 1, G, countEvals, pbest.csetN, elapsed_ms);
                        fprintf(fcrate, "%d\n", pbest.csetN);
                    }
                    if (pbest.csetN == apmin) {
                        break;
                    }
                }

                clock_t trial_end = clock();
                total_trial_time_ticks += (trial_end - trial_start);

                countResult += pbest.csetN;
                if (gBest < pbest.csetN) gBest = pbest.csetN;
                if (gWorst > pbest.csetN) gWorst = pbest.csetN;
                avgEval += countEvals;

                double trial_time_ms = ((double)(trial_end - trial_start)) / CLOCKS_PER_SEC * 1000.0;
                printf("Case %d, Trial %d:\tG=%d\tEvals=%d\tcsetN=%d\tTime=%.2fms\n",
                       current_case.No, trialNum + 1, G, countEvals, pbest.csetN, trial_time_ms);
                fprintf(finst, "%d\t%d\t%d\t%d\t%.2f\n", current_case.No, G, countEvals, pbest.csetN, trial_time_ms);
            }

            clock_t total_end_case = clock();
            double overall_case_time_ms = ((double)(total_end_case - total_start_case)) / CLOCKS_PER_SEC * 1000.0;
            double total_trial_time_ms  = ((double)total_trial_time_ticks) / CLOCKS_PER_SEC * 1000.0;

            double mean_csetN = (double)countResult / TRIALNUM;
            double average_sensors_per_subset = (double)N / mean_csetN;

            fprintf(fsummary, "%d\t%.1f\t%d\t%.2f\t%.2f\t%.2f\n",
                    current_case.No, R, current_case.N, mean_csetN, average_sensors_per_subset, overall_case_time_ms);
            printf("Case %d Summary:\n", current_case.No);
            printf("R = %.1f, N = %d\n", R, current_case.N);
            printf("Means (Average Subsets) = %.2f\n", mean_csetN);
            printf("Average Sensors per Subset = %.2f\n", average_sensors_per_subset);
            printf("Total Trial Time (ms) = %.2f\n", total_trial_time_ms);
            printf("Overall Case Time (ms) = %.2f\n\n", overall_case_time_ms);

            fclose(finst);
            fclose(favgtotal);
            fclose(fcrate);
        } else {
            // 部署失败
            fprintf(fsummary, "%d\t%.1f\t%d\tFailed\tFailed\tFailed\n",
                    current_case.No, current_case.R, current_case.N);
            printf("Case %d Deployment Failed.\n\n", current_case.No);
        }

        clock_t total_end_case = clock();
        double total_time_case_ms = ((double)(total_end_case - total_start_case)) / CLOCKS_PER_SEC * 1000.0;

        // 重置全局
        memset(OPos,    0, sizeof(OPos));
        memset(OPosf,   0, sizeof(OPosf));
        memset(field,   0, sizeof(field));
        memset(grid,    0, sizeof(grid));
        memset(fieldIndex, 0, sizeof(fieldIndex));
        memset(P,       0, sizeof(P));
        memset(pbest.d, 0, sizeof(pbest.d));
        pbest.csetN   = 0;
        pbest.fitness = 0.0;
    }

    fclose(fsummary);
    return 0;
}
