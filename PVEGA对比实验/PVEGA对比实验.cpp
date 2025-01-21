#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <random>
#include <limits>
#include <climits>
#include <string>
#include <iomanip>
#include <bitset>
#include <omp.h> // 使用OpenMP进行并行化

using namespace std;

//---------------------------------------
// 参数设置（可根据需要调节）
//---------------------------------------
const double L_AREA = 50.0;         // 区域长度（默认50，可在不同案例中随意变）
const double W_AREA = 50.0;         // 区域宽度（默认50，可在不同案例中随意变）
static const int    GRID_LEVEL = 4; // 网格分割级别：R_e / GRID_LEVEL => 单次步长

// 下列参数保持原本逻辑或仅做少量调整
const int    CMAX = GRID_LEVEL * GRID_LEVEL * 4;                // 根据R_e设定足够大的值
const int    MAX_ITER = 50;          // 每阶段GA最大迭代次数
const double FITNESS_THRESHOLD = 1e-6;  // 适应度阈值，调整为更严格的值
const int    POPULATION_SIZE = 20;       // 种群大小
const double CROSSOVER_RATE = 0.8;      // 交叉率
const double MUTATION_RATE = 0.05;      // 变异率
const int    MAX_REDUNDANCY_TRIES = 20; // 冗余调度重复尝试次数
const int    UP = 20;                  // 终止上限值

//--------------------------
// 结构体定义
//--------------------------
struct Sensor {
    int id;
    double x, y;
    double energy;
};

struct Grid {
    int id;
    double x_min;
    double x_max;
    double y_min;
    double y_max;
};

struct Chromosome {
    vector<int>  genes;     // 基因序列，0表示休眠，>=1表示激活于特定子集
    vector<char> fixedGene; // 0 表示可变，1 表示固定
    double fitness;
};

struct CaseInfo {
    int    No;
    double R;
    int    N;
};

//--------------------------
// 全局变量
//--------------------------
vector<Sensor> originalSensors; // 原始传感器
vector<Sensor> subSensors;      // 拆分后的子传感器

// 网格数组
vector<Grid> grids;

// gridCoveringSensors[grid_id] = { sensor_ids... }
vector<vector<int>> gridCoveringSensors;

// 关键区域相关变量
int apmin; // 最少覆盖次数
vector<vector<int>> criticalRegions;    // 当前关键区域集合
vector<vector<int>> allCriticalRegions; // 全部关键区域集合（可选，用于统计）

//---------------------------------------
// 距离计算
//---------------------------------------
inline double distanceEuclid(double x1, double y1, double x2, double y2) {
    double dx = x1 - x2;
    double dy = y1 - y2;
    return sqrt(dx * dx + dy * dy);
}

//---------------------------------------
// 读取传感器数据
//---------------------------------------
void readSensors(const string& filename) {
    originalSensors.clear();
    ifstream infile(filename);
    if (!infile.is_open()) {
        cerr << "无法打开文件: " << filename << endl;
        exit(1);
    }
    double x, y, energy;
    int id = 0;
    while (infile >> x >> y >> energy) {
        Sensor s;
        s.id = id++;
        s.x = x;
        s.y = y;
        s.energy = energy;
        originalSensors.push_back(s);
        if (id > 999999) break; // 防止极端大文件
    }
    infile.close();
    cout << "原始传感器数量：" << originalSensors.size() << endl;
}

//---------------------------------------
// 拆分传感器（把能量为E的物理传感器拆成 E 个子传感器，每个子传感器能量=1）
//---------------------------------------
void splitSensors() {
    subSensors.clear();
    int subId = 0;
    for (auto& os : originalSensors) {
        int num_sub = (int)ceil(os.energy);
        for (int j = 1; j <= num_sub; j++) {
            Sensor ss;
            ss.id = subId++;
            ss.x = os.x;
            ss.y = os.y;
            ss.energy = 1.0; // 子传感器的能量=1
            subSensors.push_back(ss);
        }
    }
    cout << "子传感器数量：" << subSensors.size() << endl;
}

//---------------------------------------
// 动态初始化网格 (改成和原代码类似的方式)
//---------------------------------------
// 原先是 step_default=1.0, GRID_L=50, GRID_W=50 固定；
// 现在改为根据 R_e 动态计算网格数量和步长。
void initializeGrids(double R_e) {
    // 1) 先根据 R_e/GRID_LEVEL 得到一个初步 cell_step
    double cell_step = R_e / GRID_LEVEL;
    if (cell_step < 0.0001) {
        cell_step = 0.0001;  // 防止过小
    }

    // 2) 计算网格数量
    int gridCountX = (int)floor(L_AREA / cell_step);
    int gridCountY = (int)floor(W_AREA / cell_step);
    if (gridCountX < 1) gridCountX = 1;
    if (gridCountY < 1) gridCountY = 1;

    // 3) 反算出真正步长 stepL, stepW
    double stepL = L_AREA / gridCountX;
    double stepW = W_AREA / gridCountY;

    // 4) 构建网格
    grids.clear();
    grids.reserve(gridCountX * gridCountY);

    for (int ix = 0; ix < gridCountX; ix++) {
        for (int iy = 0; iy < gridCountY; iy++) {
            Grid gr;
            gr.id = (int)grids.size();
            gr.x_min = ix * stepL;
            gr.x_max = (ix + 1) * stepL;
            if (gr.x_max > L_AREA) gr.x_max = L_AREA;

            gr.y_min = iy * stepW;
            gr.y_max = (iy + 1) * stepW;
            if (gr.y_max > W_AREA) gr.y_max = W_AREA;

            grids.push_back(gr);
        }
    }

    cout << "动态网格划分: gridCountX=" << gridCountX
        << ", gridCountY=" << gridCountY
        << ", 总网格数=" << grids.size() << endl;

    // 重置 gridCoveringSensors
    gridCoveringSensors.clear();
    gridCoveringSensors.resize(grids.size());

    // 5) 遍历每个子传感器，判断它覆盖哪些网格
    //    使用网格中心点与传感器距离 <= R_e 判断是否覆盖
#pragma omp parallel for
    for (size_t si = 0; si < subSensors.size(); si++) {
        const Sensor& s = subSensors[si];

        // 可能的网格范围 (略放大)
        // 你可以根据 R_e + 1 计算 i0, i1, j0, j1, 或者干脆遍历全部
        // 这里只是示例做法
        double left = (s.x - R_e), right = (s.x + R_e);
        double bottom = (s.y - R_e), top = (s.y + R_e);

        // 限制在 [0, L_AREA], [0, W_AREA]
        if (left < 0.0) left = 0.0;
        if (right > L_AREA) right = L_AREA;
        if (bottom < 0.0) bottom = 0.0;
        if (top > W_AREA) top = W_AREA;

        // 计算对应的网格索引范围
        int min_ix = (int)floor(left / stepL);
        int max_ix = (int)floor(right / stepL);
        int min_iy = (int)floor(bottom / stepW);
        int max_iy = (int)floor(top / stepW);

        if (min_ix < 0) min_ix = 0;
        if (max_ix >= gridCountX) max_ix = gridCountX - 1;
        if (min_iy < 0) min_iy = 0;
        if (max_iy >= gridCountY) max_iy = gridCountY - 1;

        // 逐格判断
        for (int ix = min_ix; ix <= max_ix; ix++) {
            for (int iy = min_iy; iy <= max_iy; iy++) {
                int gi = ix * gridCountY + iy; // grids[] 的索引
                // 以网格中心为准
                double cx = (grids[gi].x_min + grids[gi].x_max) * 0.5;
                double cy = (grids[gi].y_min + grids[gi].y_max) * 0.5;

                if (distanceEuclid(cx, cy, s.x, s.y) <= R_e) {
#pragma omp critical
                    {
                        gridCoveringSensors[gi].push_back((int)si);
                    }
                }
            }
        }
    }
}

//---------------------------------------
// 输出关键区域信息
//---------------------------------------
void outputCriticalRegionsInfo(const vector<vector<int>>& regions) {
    // 输出关键区域的总数量
    cout << "关键区域总数量：" << regions.size() << endl;

    // 输出每个关键区域中传感器的数量
    for (size_t i = 0; i < regions.size(); i++) {
        cout << "关键区域 " << i + 1 << " 包含 " << regions[i].size() << " 个传感器。" << endl;
    }

    // 计算所有关键区域中传感器的总数量（包含重复）
    int totalSensorsInCriticalRegions = 0;
    for (const auto& region : regions) {
        totalSensorsInCriticalRegions += (int)region.size();
    }
    cout << "所有关键区域中传感器的总数量（包含重复）： " << totalSensorsInCriticalRegions << endl;

    // 计算所有关键区域中唯一传感器的总数
    vector<char> sensorUsed(subSensors.size(), 0);
    int uniqueCriticalSensors = 0;
    for (const auto& region : regions) {
        for (auto sensor_id : region) {
            if (!sensorUsed[sensor_id]) {
                sensorUsed[sensor_id] = 1;
                uniqueCriticalSensors++;
            }
        }
    }

    // 输出最大可能生成的子覆盖集数量（理论值）
    cout << "最大可能生成的子覆盖集数量（理论上等于关键区域总数量）： " << uniqueCriticalSensors << endl;
}

//---------------------------------------
// 识别关键区域
//---------------------------------------
void identifyCriticalRegions(const vector<int>& genes, int currentSubset) {
    criticalRegions.clear();

    // 统计每个网格的覆盖次数
    vector<int> coverageCount(grids.size(), 0);

#pragma omp parallel for
    for (size_t gi = 0; gi < grids.size(); gi++) {
        int count = 0;
        for (auto sensor_id : gridCoveringSensors[gi]) {
            if ((size_t)sensor_id < genes.size() && genes[sensor_id] == currentSubset) {
                count++;
            }
        }
        coverageCount[gi] = count;
    }

    // 找到最小覆盖次数
    apmin = INT_MAX;
    for (auto c : coverageCount) {
        if (c > 0 && c < apmin) { // 忽略未覆盖的网格
            apmin = c;
        }
    }

    // 识别所有覆盖次数==apmin的网格作为关键区域
    for (size_t gi = 0; gi < grids.size(); gi++) {
        if (coverageCount[gi] == apmin) {
            // 记录覆盖该网格的所有传感器
            vector<int> coveringSensors;
            for (auto sensor_id : gridCoveringSensors[gi]) {
                if ((size_t)sensor_id < genes.size() && genes[sensor_id] == currentSubset) {
                    coveringSensors.push_back(sensor_id);
                }
            }
            criticalRegions.push_back(coveringSensors);
            allCriticalRegions.push_back(coveringSensors); // 记录到全局关键区域
        }
    }
}

//---------------------------------------
// 检查当前已构建子集能否覆盖所有网格
//---------------------------------------
bool allCovered(const vector<int>& genes, int currentSubset) {
    for (size_t gi = 0; gi < grids.size(); gi++) {
        bool covered = false;
        for (auto sensor_id : gridCoveringSensors[gi]) {
            if ((size_t)sensor_id < genes.size() && genes[sensor_id] == currentSubset) {
                covered = true;
                break;
            }
        }
        if (!covered) return false;
    }
    return true;
}

//---------------------------------------
// 快速冗余调度
//---------------------------------------
bool tryRedundancyScheduling(vector<int>& genes, int currentSubset, const vector<char>& fixedGene,
    int n_max_size, bool flag = true, int k = 2)
{
    bool anySleepDone = false;
    int n = n_max_size;

    while (n >= 0) {
        int sleep_count = (int)pow(k, n);
        bool sleepSuccess = false;

        for (int attempt = 1; attempt <= MAX_REDUNDANCY_TRIES; attempt++) {
            // 选可休眠的子传感器
            vector<int> candidates;
            for (size_t i = 0; i < genes.size(); i++) {
                if (!fixedGene[i] && genes[i] == currentSubset) {
                    candidates.push_back((int)i);
                }
            }
            if ((int)candidates.size() < sleep_count) continue;

            // Shuffle
            static mt19937 g(random_device{}());
            shuffle(candidates.begin(), candidates.end(), g);

            // 让前 sleep_count 个休眠
            vector<int> to_sleep(candidates.begin(), candidates.begin() + sleep_count);

            // 备份
            vector<int> backup = genes;
            // 休眠
            for (int idx : to_sleep) {
                genes[idx] = 0;
            }

            // 覆盖性检查
            if (allCovered(genes, currentSubset)) {
                sleepSuccess = true;
                anySleepDone = true;
                break;
            }
            // 恢复
            genes = backup;
        }
        if (!flag) break;
        if (sleepSuccess) {
            // 成功后尝试更大休眠
            n = n_max_size;
        }
        else {
            // 无法休眠
            n--;
            if (n == 0) break;
        }
    }
    return anySleepDone;
}

//---------------------------------------
// 关键区域调度
//---------------------------------------
bool keyRegionScheduling(vector<int>& genes, int currentSubset, const vector<char>& fixedGene, int n_max_size) {
    // 将关键区域内的多余节点调度至休眠状态
    return tryRedundancyScheduling(genes, currentSubset, fixedGene, n_max_size);
}

//---------------------------------------
// 适应度评估
//---------------------------------------
void evaluateFitness(vector<Chromosome>& population, int currentSubset) {
#pragma omp parallel for
    for (size_t p = 0; p < population.size(); p++) {
        Chromosome& chr = population[p];
        if (allCovered(chr.genes, currentSubset)) {
            // 计算活跃传感器数量
            int sum_genes = 0;
            for (auto gene : chr.genes) {
                if (gene == currentSubset) sum_genes++;
            }
            chr.fitness = (sum_genes > 0) ? (1.0 / (double)sum_genes) : 0.0;
        }
        else {
            chr.fitness = 0.0;
        }
    }
}

//---------------------------------------
// 选择操作 (轮盘赌)
//---------------------------------------
Chromosome selectParent(const vector<Chromosome>& pop, mt19937& gen) {
    double total = 0.0;
    for (const auto& c : pop) total += c.fitness;
    if (total <= 1e-14) {
        // 随机返回
        uniform_int_distribution<> d(0, (int)pop.size() - 1);
        return pop[d(gen)];
    }
    uniform_real_distribution<> dis(0.0, total);
    double r = dis(gen);
    double cum = 0.0;
    for (const auto& c : pop) {
        cum += c.fitness;
        if (cum >= r) return c;
    }
    return pop.back();
}

//---------------------------------------
// 交叉操作
//---------------------------------------
pair<Chromosome, Chromosome> crossover(const Chromosome& p1, const Chromosome& p2, mt19937& gen) {
    Chromosome o1 = p1, o2 = p2;
    uniform_real_distribution<> dis(0.0, 1.0);
    if (dis(gen) < CROSSOVER_RATE) {
        uniform_int_distribution<> point_dis(0, (int)p1.genes.size() - 1);
        int point = point_dis(gen);
        for (size_t i = point; i < p1.genes.size(); i++) {
            if (!p1.fixedGene[i] && !p2.fixedGene[i]) {
                swap(o1.genes[i], o2.genes[i]);
            }
        }
    }
    return make_pair(o1, o2);
}

//---------------------------------------
// 变异操作
//---------------------------------------
void mutate(Chromosome& chr, mt19937& gen) {
    uniform_real_distribution<> dis(0.0, 1.0);
    for (size_t i = 0; i < chr.genes.size(); i++) {
        if (!chr.fixedGene[i]) {
            if (dis(gen) < MUTATION_RATE) {
                chr.genes[i] = 0; // 休眠
            }
        }
    }
}

//---------------------------------------
// 遗传算法构建当前子集
//---------------------------------------
bool runGeneticAlgorithm(Chromosome& baseChrom, int currentSubset) {
    // 1) 冗余调度
    int n_max_size = (int)floor(log2(subSensors.size()));
    bool redundancySuccess = tryRedundancyScheduling(baseChrom.genes, currentSubset, baseChrom.fixedGene, n_max_size);

    // 2) 初始化种群
    vector<Chromosome> population;
    population.reserve(POPULATION_SIZE);

    mt19937 gen(random_device{}());

    for (int i = 0; i < POPULATION_SIZE; i++) {
        Chromosome chr = baseChrom;
        // 增加随机休眠操作
        redundancySuccess = tryRedundancyScheduling(chr.genes, currentSubset, chr.fixedGene, 2, false, 1);
        if (redundancySuccess) {
            keyRegionScheduling(chr.genes, currentSubset, chr.fixedGene, 2);
        }
        // 确保关键区域唯一覆盖
        identifyCriticalRegions(chr.genes, currentSubset);
        bool uniqueCoverage = true;
        for (const auto& region : criticalRegions) {
            int activeSensors = 0;
            for (auto sid : region) {
                if ((size_t)sid < chr.genes.size() && chr.genes[sid] == currentSubset) {
                    activeSensors++;
                    if (activeSensors > 1) {
                        uniqueCoverage = false;
                        break;
                    }
                }
            }
            if (!uniqueCoverage) break;
        }
        if (!uniqueCoverage) {
            // 简单处理
            for (const auto& region : criticalRegions) {
                if (!region.empty()) {
                    int sensor_id = region[0];
                    chr.genes[sensor_id] = currentSubset;
                }
            }
        }
        population.push_back(chr);
    }

    evaluateFitness(population, currentSubset);

    // 3) 寻找当前最好
    Chromosome best;
    best.fitness = 0.0;

    int iteration = 0;
    double lastBest = 0.0;
    int stableCount = 0;

    // 4) 主循环
    while (iteration < MAX_ITER) {
        // 找到最优
        for (const auto& c : population) {
            if (c.fitness > best.fitness) best = c;
        }

        // 检查阈值
        if (best.fitness >= FITNESS_THRESHOLD) break;

        // 检查是否稳定
        if (fabs(best.fitness - lastBest) < 1e-9) {
            stableCount++;
            if (stableCount > UP) break;
        }
        else {
            stableCount = 0;
            lastBest = best.fitness;
        }

        // 生成下一代
        vector<Chromosome> newPop;
        newPop.reserve(POPULATION_SIZE);
        while ((int)newPop.size() < POPULATION_SIZE) {
            Chromosome p1 = selectParent(population, gen);
            Chromosome p2 = selectParent(population, gen);
            auto off = crossover(p1, p2, gen);
            mutate(off.first, gen);
            mutate(off.second, gen);
            // 前向调度
            redundancySuccess = tryRedundancyScheduling(off.first.genes, currentSubset, off.first.fixedGene, 2, false, 1);
            newPop.push_back(off.first);
            if ((int)newPop.size() < POPULATION_SIZE)
                newPop.push_back(off.second);
        }

        evaluateFitness(newPop, currentSubset);

        // 合并并排序
        vector<Chromosome> combined(population);
        combined.insert(combined.end(), newPop.begin(), newPop.end());
        sort(combined.begin(), combined.end(),
            [](const Chromosome& a, const Chromosome& b) { return a.fitness > b.fitness; });
        // 保留前POPSIZE
        if (combined.size() > POPULATION_SIZE) {
            combined.resize(POPULATION_SIZE);
        }
        population = combined;

        evaluateFitness(population, currentSubset);
        iteration++;
    }

    // 再次确认最好
    for (const auto& c : population) {
        if (c.fitness > best.fitness) best = c;
    }

    // 若找到可行解
    if (best.fitness > 0.0) {
        baseChrom = best;
        // 确保关键区域唯一覆盖
        identifyCriticalRegions(baseChrom.genes, currentSubset);
        bool uniqueCoverage1 = true;
        for (const auto& region : criticalRegions) {
            int activeSensors = 0;
            for (auto sid : region) {
                if ((size_t)sid < baseChrom.genes.size() && baseChrom.genes[sid] == currentSubset) {
                    activeSensors++;
                    if (activeSensors > 1) {
                        uniqueCoverage1 = false;
                        break;
                    }
                }
            }
            if (!uniqueCoverage1) break;
        }
        if (!uniqueCoverage1) {
            for (const auto& region : criticalRegions) {
                if (!region.empty()) {
                    int sensor_id = region[0];
                    baseChrom.genes[sensor_id] = currentSubset;
                }
            }
        }
        return true;
    }
    return false;
}

//---------------------------------------
// 主函数
//---------------------------------------
int main() {
    // 定义所有案例
    vector<CaseInfo> cases = {
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

    // 打开总结文件
    ofstream fsummary("summary.txt");
    if (!fsummary.is_open()) {
        cerr << "无法创建 summary.txt 文件。" << endl;
        return 1;
    }

    // 写入表头
    fsummary << "No.\tR\tN\tMeans\taverage\tTimes(ms)\n";

    // 遍历每个案例
    for (const auto& current_case : cases) {
        cout << "---------------------------------------" << endl;
        cout << "处理案例 " << current_case.No << ": R=" << current_case.R
            << ", N=" << current_case.N << endl;

        // 重置全局变量
        originalSensors.clear();
        subSensors.clear();
        grids.clear();
        gridCoveringSensors.clear();
        criticalRegions.clear();
        allCriticalRegions.clear();

        // 设置输入文件名
        string input_filename = "input_" + to_string(current_case.N) + ".txt";

        // 读取并拆分传感器
        readSensors(input_filename);
        splitSensors();

        // 使用动态网格初始化
        initializeGrids(current_case.R);

        // 输出初始信息
        cout << "原始传感器数量：" << originalSensors.size() << endl;
        cout << "子传感器数量：" << subSensors.size() << endl;
        cout << "网格数量：" << grids.size() << endl;

        // 初始化染色体：全部休眠
        Chromosome baseChrom;
        baseChrom.genes.resize(subSensors.size(), 0);
        baseChrom.fixedGene.resize(subSensors.size(), 0);
        baseChrom.fitness = 0.0;

        int currentSubset = 1;
        vector<vector<int>> finalSubsets;

        // 输出初始关键区域信息（所有传感器均激活）
        {
            vector<int> allActiveGenes(subSensors.size(), currentSubset);
            allCriticalRegions.clear();
            identifyCriticalRegions(allActiveGenes, currentSubset);
            vector<vector<int>> initialCriticalRegions = criticalRegions;
            outputCriticalRegionsInfo(initialCriticalRegions);
        }

        // 记录案例开始时间
        clock_t case_start = clock();

        // 多子集构建循环
        while (true) {
            // 若不是第一个子集，表示之前子集已固定
            // 将所有当前为0的未固定基因设为currentSubset
            bool anyAvailable = false;
            for (size_t i = 0; i < baseChrom.genes.size(); i++) {
                if (baseChrom.genes[i] == 0 && !baseChrom.fixedGene[i]) {
                    baseChrom.genes[i] = currentSubset;
                    anyAvailable = true;
                }
            }
            if (!anyAvailable) {
                // 没有可用子传感器，退出
                break;
            }

            // 运行GA
            if (!runGeneticAlgorithm(baseChrom, currentSubset)) {
                // 无法构建新的覆盖子集，还原
                for (size_t i = 0; i < baseChrom.genes.size(); i++) {
                    if (baseChrom.genes[i] == currentSubset && !baseChrom.fixedGene[i]) {
                        baseChrom.genes[i] = 0;
                    }
                }
                break;
            }

            // 再次检查
            if (!allCovered(baseChrom.genes, currentSubset)) {
                for (size_t i = 0; i < baseChrom.genes.size(); i++) {
                    if (baseChrom.genes[i] == currentSubset && !baseChrom.fixedGene[i]) {
                        baseChrom.genes[i] = 0;
                    }
                }
                break;
            }

            // 收集当前子集索引
            vector<int> subsetIndices;
            for (size_t i = 0; i < baseChrom.genes.size(); i++) {
                if (baseChrom.genes[i] == currentSubset) {
                    subsetIndices.push_back((int)i);
                }
            }
            if (subsetIndices.empty()) {
                // 子集为空还原并退出
                for (size_t i = 0; i < baseChrom.genes.size(); i++) {
                    if (baseChrom.genes[i] == currentSubset && !baseChrom.fixedGene[i]) {
                        baseChrom.genes[i] = 0;
                    }
                }
                break;
            }

            // 固定之
            for (auto idx : subsetIndices) {
                baseChrom.fixedGene[idx] = 1;
            }

            // 输出子集信息
            cout << "子集" << currentSubset << "(";
            for (size_t i = 0; i < subsetIndices.size(); i++) {
                int originalSensorId = 0;
                int subSensorIndex = subsetIndices[i];
                // 找到子传感器属于哪个原始传感器
                int cumulative = 0;
                while (originalSensorId < (int)originalSensors.size()) {
                    int num_sub = (int)ceil(originalSensors[originalSensorId].energy);
                    if (subSensorIndex < cumulative + num_sub) {
                        break;
                    }
                    cumulative += num_sub;
                    originalSensorId++;
                }
                if (originalSensorId < (int)originalSensors.size()) {
                    int subId = subSensorIndex - cumulative + 1;
                    cout << "s_" << (originalSensorId + 1) << "_" << subId;
                }
                else {
                    cout << "s_" << finalSubsets.size() + 1 << "_" << subSensorIndex + 1;
                }
                if (i < subsetIndices.size() - 1) cout << ", ";
            }
            cout << ")，包含 " << subsetIndices.size() << " 个传感器。" << endl;

            finalSubsets.push_back(subsetIndices);
            currentSubset++;
        }

        // 结束计时
        clock_t case_end = clock();
        double elapsed_time = (double)(case_end - case_start) / CLOCKS_PER_SEC * 1000.0;

        // 计算统计数据
        double mean_csetN = (double)finalSubsets.size();
        double total_sensors_in_subsets = 0.0;
        for (const auto& subset : finalSubsets) {
            total_sensors_in_subsets += subset.size();
        }
        double average_sensors_per_subset = 0.0;
        if (mean_csetN > 0.0) {
            average_sensors_per_subset = total_sensors_in_subsets / mean_csetN;
        }

        // 写入总结文件
        fsummary << current_case.No << "\t"
            << fixed << setprecision(1) << current_case.R << "\t"
            << current_case.N << "\t"
            << fixed << setprecision(2) << mean_csetN << "\t"
            << fixed << setprecision(2) << average_sensors_per_subset << "\t"
            << fixed << setprecision(1) << elapsed_time << "\n";

        // 创建覆盖集输出文件
        int N = current_case.N;
        int R_int = static_cast<int>(current_case.R);
        string coverage_filename = "output_" + to_string(N) + "_" + to_string(R_int) + ".txt";
        ofstream fcoverage(coverage_filename);
        if (!fcoverage.is_open()) {
            cerr << "无法创建文件: " << coverage_filename << endl;
            return 1;
        }

        // 写入子覆盖集信息
        for (size_t k = 0; k < finalSubsets.size(); k++) {
            fcoverage << (k + 1) << ", (";
            for (size_t i = 0; i < finalSubsets[k].size(); i++) {
                int originalSensorId = 0;
                int subSensorIndex = finalSubsets[k][i];
                int cumulative = 0;
                while (originalSensorId < (int)originalSensors.size()) {
                    int num_sub = (int)ceil(originalSensors[originalSensorId].energy);
                    if (subSensorIndex < cumulative + num_sub) {
                        break;
                    }
                    cumulative += num_sub;
                    originalSensorId++;
                }
                if (originalSensorId < (int)originalSensors.size()) {
                    int subId = subSensorIndex - cumulative + 1;
                    fcoverage << "s_" << (originalSensorId + 1) << "_" << subId;
                }
                else {
                    fcoverage << "s_" << finalSubsets[k].size() + 1 << "_" << subSensorIndex + 1;
                }
                if (i < finalSubsets[k].size() - 1) fcoverage << ", ";
            }
            fcoverage << ")" << endl;
        }
        fcoverage << endl;
        fcoverage.close();

        // 控制台输出
        if (!finalSubsets.empty()) {
            cout << "成功构建 " << finalSubsets.size() << " 个子集：" << endl;
            for (size_t k = 0; k < finalSubsets.size(); k++) {
                cout << "子集" << k + 1 << "(";
                for (size_t i = 0; i < finalSubsets[k].size(); i++) {
                    int originalSensorId = 0;
                    int subSensorIndex = finalSubsets[k][i];
                    int cumulative = 0;
                    while (originalSensorId < (int)originalSensors.size()) {
                        int num_sub = (int)ceil(originalSensors[originalSensorId].energy);
                        if (subSensorIndex < cumulative + num_sub) {
                            break;
                        }
                        cumulative += num_sub;
                        originalSensorId++;
                    }
                    if (originalSensorId < (int)originalSensors.size()) {
                        int subId = subSensorIndex - cumulative + 1;
                        cout << "s_" << (originalSensorId + 1) << "_" << subId;
                    }
                    else {
                        cout << "s_" << finalSubsets[k].size() + 1 << "_" << subSensorIndex + 1;
                    }
                    if (i < finalSubsets[k].size() - 1) cout << ", ";
                }
                cout << ")，包含 " << finalSubsets[k].size() << " 个传感器。" << endl;
            }
            cout << "总的生命周期（子集数）:" << finalSubsets.size() << endl;
        }
        else {
            cout << "未能构建满足要求的覆盖子集。" << endl;
        }

        cout << "Elapsed time: " << fixed << setprecision(2)
            << elapsed_time << " ms" << endl;
        cout << "---------------------------------------" << endl << endl;
    }

    fsummary.close();
    return 0;
}
