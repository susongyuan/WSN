#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <algorithm>
#include <limits>
#include <cmath>
#include <chrono>
#include <iomanip>

using namespace std;

// ---------------------------
// 物理传感器结构体
// ---------------------------
struct PhysicalSensor {
    int id;         // 物理传感器ID（0-based）
    double x;       // x坐标
    double y;       // y坐标
    int energy;     // 传感器能量，可以理解为能工作多少个时间单位
};

// ---------------------------
// 覆盖集结构体
// ---------------------------
struct CoverageSet {
    int id;                  // 覆盖集ID（从文件读取，可能不是0-based）
    vector<int> sensorIDs;   // 包含的物理传感器ID(0-based)
};

// ---------------------------
// 调度结果
// ---------------------------
struct SchedulingResult {
    vector<int> schedule; // 记录覆盖集ID（可重复），每次使用覆盖集则push_back一次
    int lifetime;         // 网络生命期 (总共覆盖了多少时间单位)
};

// ---------------------------
// 读取传感器
// ---------------------------
vector<PhysicalSensor> readSensors(const string& filename) {
    ifstream fin(filename);
    if (!fin.is_open()) {
        cerr << "无法打开传感器文件: " << filename << endl;
        exit(1);
    }
    vector<PhysicalSensor> sensors;
    double xx, yy, ee;
    int idx = 0;
    while (fin >> xx >> yy >> ee) {
        PhysicalSensor ps;
        ps.id = idx; // 0-based
        ps.x = xx;
        ps.y = yy;
        ps.energy = (int)floor(ee);
        if (ps.energy < 0) ps.energy = 0;
        sensors.push_back(ps);
        idx++;
    }
    fin.close();
    return sensors;
}

// ---------------------------
// 读取覆盖集
// ---------------------------
vector<CoverageSet> readCoverageSets(const string& filename) {
    ifstream fin(filename);
    if (!fin.is_open()) {
        cerr << "无法打开覆盖集文件: " << filename << endl;
        exit(1);
    }
    vector<CoverageSet> covers;
    string line;
    while (getline(fin, line)) {
        if (line.empty()) continue;
        auto commaPos = line.find(',');
        if (commaPos == string::npos) {
            cerr << "覆盖集格式错误: " << line << endl;
            continue;
        }
        string idStr = line.substr(0, commaPos);
        idStr.erase(remove_if(idStr.begin(), idStr.end(), ::isspace), idStr.end());
        int cID = stoi(idStr);

        auto lParen = line.find('(');
        auto rParen = line.find(')');
        if (lParen == string::npos || rParen == string::npos || rParen <= lParen) {
            cerr << "覆盖集格式错误(括号缺失): " << line << endl;
            continue;
        }
        string inside = line.substr(lParen + 1, rParen - (lParen + 1));
        vector<string> tokens;
        {
            stringstream ss(inside);
            string token;
            while (getline(ss, token, ',')) {
                token.erase(remove_if(token.begin(), token.end(), ::isspace), token.end());
                if (!token.empty())
                    tokens.push_back(token);
            }
        }
        CoverageSet cov;
        cov.id = cID;
        set<int> uniqueSensors;
        for (auto& tk : tokens) {
            if (tk.size() < 3 || tk.substr(0, 2) != "s_") continue;
            string tail = tk.substr(2);
            auto underscorePos = tail.find('_');
            if (underscorePos == string::npos) continue;
            string sensorIDStr = tail.substr(0, underscorePos);
            int physicalID_1based = stoi(sensorIDStr);
            int physicalID = physicalID_1based - 1;
            if (physicalID < 0) {
                cerr << "警告: 出现无效物理传感器ID: " << tk << endl;
                continue;
            }
            uniqueSensors.insert(physicalID);
        }
        cov.sensorIDs.assign(uniqueSensors.begin(), uniqueSensors.end());
        covers.push_back(cov);
    }
    fin.close();
    return covers;
}

// -----------------------------------------------------------------
// 3) Greedy_MCSS：一次只使用覆盖集1个时间单位
// -----------------------------------------------------------------
SchedulingResult Greedy_MCSS(vector<PhysicalSensor>& sensors,
                             vector<CoverageSet>& coverageSets)
{
    SchedulingResult result;
    result.lifetime = 0;  // 防止未初始化

    while (true) {
        int chosenIndex = -1;
        int minEnergyInSet = numeric_limits<int>::max();

        // (a) 找到最小传感器能量的覆盖集
        for (int i = 0; i < (int)coverageSets.size(); i++) {
            const auto &cs = coverageSets[i];
            int localMin = numeric_limits<int>::max();
            bool allPositive = true;
            for (int sid : cs.sensorIDs) {
                if (sensors[sid].energy <= 0) {
                    allPositive = false;
                    break;
                }
                localMin = min(localMin, sensors[sid].energy);
            }
            if (allPositive && localMin < minEnergyInSet) {
                minEnergyInSet = localMin;
                chosenIndex = i;
            }
        }

        // (b) 如果找不到可用覆盖集，则退出
        if (chosenIndex < 0) {
            break;
        }

        // (c) 使用该覆盖集一次（1个时间单位）
        int chosenCoverID = coverageSets[chosenIndex].id;
        result.schedule.push_back(chosenCoverID);
        result.lifetime++;

        // 减能量
        for (int sid : coverageSets[chosenIndex].sensorIDs) {
            sensors[sid].energy--;
        }

        // (d) 移除能量耗尽传感器涉及的覆盖集
        vector<int> zeroSensors;
        for (auto &ps : sensors) {
            if (ps.energy <= 0) {
                zeroSensors.push_back(ps.id);
            }
        }
        coverageSets.erase(
            remove_if(coverageSets.begin(), coverageSets.end(),
                      [&](const CoverageSet &cs){
                          for (int zsid : zeroSensors) {
                              if (find(cs.sensorIDs.begin(), cs.sensorIDs.end(), zsid) != cs.sensorIDs.end()) {
                                  return true;
                              }
                          }
                          return false;
                      }),
            coverageSets.end()
        );
    }
    return result;
}

// ----------------------------------------------------------------------
// 4) 多阶段调度：MCSSA
//    每调度一次覆盖集就相当于消耗1个时间单位，而不是一次性加好几次
// ----------------------------------------------------------------------
SchedulingResult MCSSA(vector<PhysicalSensor>& sensors,
                       vector<CoverageSet>& coverageSets)
{
    // 阶段1: 初始化
    map<int, vector<int>> Si;
    set<int> S;    // 活动传感器
    set<int> CS;   // 候选传感器

    for (int i = 0; i < (int)sensors.size(); i++) {
        PhysicalSensor &ps = sensors[i];
        S.insert(i);

        // 找到所有包含 sensor i 的覆盖集ID
        for (auto &cs : coverageSets) {
            if (find(cs.sensorIDs.begin(), cs.sensorIDs.end(), i) != cs.sensorIDs.end()) {
                Si[i].push_back(cs.id);
            }
        }
        // 如果该传感器不在任何覆盖集中，则从 S 中删除
        if (Si[i].empty()) {
            S.erase(i);
        }
        // 若传感器能量 < 覆盖集数量，则放入 CS
        else if (ps.energy < (int)Si[i].size()) {
            CS.insert(i);
        }
    }

    // 第1阶段结果
    vector<int> Cp1;
    int Tlife_p1 = 0;

    // 可用覆盖集(副本)
    vector<CoverageSet> availableCoverageSets = coverageSets;

    // (A) 处理候选传感器
    while (!CS.empty()) {
        // 选最小能量的候选传感器
        auto minIt = min_element(CS.begin(), CS.end(),
                                 [&](int a, int b){
                                     return sensors[a].energy < sensors[b].energy;
                                 });
        int si = *minIt;
        // 传感器 si 的能量
        if (sensors[si].energy <= 0) {
            CS.erase(si);
            S.erase(si);
            continue;
        }

        // 一次只使用 1 个时间单位
        bool used = false;
        for (int coverID : Si[si]) {
            auto it = find_if(availableCoverageSets.begin(), availableCoverageSets.end(),
                              [coverID](const CoverageSet &cset){return cset.id == coverID;});
            if (it == availableCoverageSets.end()) {
                continue;
            }
            // 检查所有传感器能量>0
            bool valid = true;
            for (int sid : it->sensorIDs) {
                if (sensors[sid].energy <= 0) {
                    valid = false;
                    break;
                }
            }
            if (valid) {
                // 使用该覆盖集
                Cp1.push_back(it->id);
                Tlife_p1++;
                for (int sid : it->sensorIDs) {
                    sensors[sid].energy--;
                }
                used = true;
                break;
            }
        }

        // 用完后，移除能量<=0的传感器所在覆盖集
        vector<int> zeroSensors;
        for (auto &ps : sensors) {
            if (ps.energy <= 0) {
                zeroSensors.push_back(ps.id);
            }
        }
        availableCoverageSets.erase(
            remove_if(availableCoverageSets.begin(), availableCoverageSets.end(),
                      [&](const CoverageSet &cs){
                          for (int zsid : zeroSensors) {
                              if (find(cs.sensorIDs.begin(), cs.sensorIDs.end(), zsid) != cs.sensorIDs.end()) {
                                  return true;
                              }
                          }
                          return false;
                      }),
            availableCoverageSets.end()
        );

        // 如果用完后传感器 si 不再符合候选条件，则从CS中移除
        if (used) {
            if (sensors[si].energy <= 0 || (int)Si[si].size() <= sensors[si].energy) {
                CS.erase(si);
            }
        } else {
            // 未使用成功，说明找不到可用覆盖集
            CS.erase(si);
        }
    }

    // (B) 第2阶段：处理剩余传感器
    vector<int> Cp2;
    int Tlife_p2 = 0;

    // 不再候选的 S 中传感器继续尝试调度
    for (auto it = S.begin(); it != S.end();) {
        int si = *it;
        int bi = sensors[si].energy;
        int siSize = (int)Si[si].size();
        if (bi <= 0 || siSize == 0) {
            it = S.erase(it);
            continue;
        }

        // 多次使用(每次1时间单位)
        bool stillValid = true;
        while (bi > 0 && stillValid) {
            stillValid = false;
            for (int coverID : Si[si]) {
                auto cIt = find_if(availableCoverageSets.begin(), availableCoverageSets.end(),
                                   [coverID](const CoverageSet &cset){return cset.id == coverID;});
                if (cIt == availableCoverageSets.end()) {
                    continue;
                }
                // 检查所有传感器能量>0
                bool valid = true;
                for (int sid : cIt->sensorIDs) {
                    if (sensors[sid].energy <= 0) {
                        valid = false;
                        break;
                    }
                }
                if (valid) {
                    // 用 1 次
                    Cp2.push_back(cIt->id);
                    Tlife_p2++;
                    for (int sid : cIt->sensorIDs) {
                        sensors[sid].energy--;
                        if (sensors[sid].energy < 0) {
                            sensors[sid].energy = 0;
                        }
                    }
                    bi--;
                    stillValid = true;
                    break; 
                }
            }
        }

        // 用完后，更新 availableCoverageSets
        vector<int> zeroSensors;
        for (auto &ps : sensors) {
            if (ps.energy <= 0) {
                zeroSensors.push_back(ps.id);
            }
        }
        availableCoverageSets.erase(
            remove_if(availableCoverageSets.begin(), availableCoverageSets.end(),
                      [&](const CoverageSet &cs){
                          for (int zsid : zeroSensors) {
                              if (find(cs.sensorIDs.begin(), cs.sensorIDs.end(), zsid) != cs.sensorIDs.end()) {
                                  return true;
                              }
                          }
                          return false;
                      }),
            availableCoverageSets.end()
        );

        sensors[si].energy = bi;
        it = S.erase(it);
    }

    // (C) 第3阶段：使用贪心子算法
    // 收集剩余可用的覆盖集
    vector<CoverageSet> remainingSets;
    for (auto &cs : availableCoverageSets) {
        bool valid = true;
        for (int sid : cs.sensorIDs) {
            if (sensors[sid].energy <= 0) {
                valid = false;
                break;
            }
        }
        if (valid) {
            remainingSets.push_back(cs);
        }
    }

    // **一定要初始化 finalRes，避免未定义值**
    SchedulingResult finalRes;
    finalRes.lifetime = 0; 
    // 即使 remainingSets 为空，也不会导致 finalRes.lifetime 未初始化

    if (!remainingSets.empty()) {
        finalRes = Greedy_MCSS(sensors, remainingSets);
    }

    vector<int> finalSchedule;
    finalSchedule.insert(finalSchedule.end(), Cp1.begin(), Cp1.end());
    finalSchedule.insert(finalSchedule.end(), Cp2.begin(), Cp2.end());
    finalSchedule.insert(finalSchedule.end(), finalRes.schedule.begin(), finalRes.schedule.end());

    int totalLife = Tlife_p1 + Tlife_p2 + finalRes.lifetime;

    return { finalSchedule, totalLife };
}

// ---------------------------
// 包一层函数：multiStageScheduling
// ---------------------------
SchedulingResult multiStageScheduling(vector<PhysicalSensor>& sensors,
                                      vector<CoverageSet> coverageSets)
{
    return MCSSA(sensors, coverageSets);
}

// -----------------------------
// main函数测试(示例)
// -----------------------------
int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    // 你可以自定义(N, R)及文件名
    vector<pair<int, double>> caseList = {
        {500, 5.0},
        {500, 10.0},
        {1000, 5.0},
        {1000, 10.0},
        {2500, 5.0},
        {2500, 10.0},
        {5000, 5.0},
        {5000, 10.0},
        {10000, 5.0},
        {10000, 10.0},
    };

    ofstream foutSum("summary.txt");
    if (!foutSum.is_open()) {
        cerr << "无法创建 summary.txt\n";
        return 1;
    }
    foutSum << "N\tR\tCoverSetsUsed\tLifetime\tAvgSensorsPerCoverSet\tTime(ms)\n";

    for (auto &c : caseList) {
        int N = c.first;
        double R = c.second;
        cout << "=========================\n";
        cout << "处理案例: N=" << N << ", R=" << R << "\n";

        string inputFile = "input_" + to_string(N) + ".txt";
        int RR = (int)R;
        string coverFile = "output_" + to_string(N) + "_" + to_string(RR) + ".txt";

        // 读取传感器
        vector<PhysicalSensor> sensors = readSensors(inputFile);
        cout << "读取传感器数量: " << sensors.size() << "\n";

        // 读取覆盖集
        vector<CoverageSet> coverageSets = readCoverageSets(coverFile);
        cout << "读取覆盖集数量: " << coverageSets.size() << "\n";

        // 计时开始
        auto t1 = chrono::high_resolution_clock::now();
        // 调度
        SchedulingResult result = multiStageScheduling(sensors, coverageSets);
        // 计时结束
        auto t2 = chrono::high_resolution_clock::now();
        double usedMS = chrono::duration<double, milli>(t2 - t1).count();

        // result.schedule.size() 就是多阶段使用覆盖集的总次数
        cout << "调度后, 共使用了 " << result.schedule.size()
             << " 次覆盖集(时间单位), 生命期 = " << result.lifetime << "\n";

        // 统计平均每次使用覆盖集时用到的传感器数
        double totalSensorsUsed = 0.0;
        for (auto cid : result.schedule) {
            auto it = find_if(coverageSets.begin(), coverageSets.end(),
                              [cid](const CoverageSet& cs) { return cs.id == cid; });
            if (it != coverageSets.end()) {
                totalSensorsUsed += it->sensorIDs.size();
            }
        }
        double avgSensorsPerCover = 0.0;
        if (!result.schedule.empty()) {
            avgSensorsPerCover = totalSensorsUsed / result.schedule.size();
        }

        // 写入 summary
        foutSum << N << "\t" << R << "\t"
                << result.schedule.size() << "\t"
                << result.lifetime << "\t"
                << fixed << setprecision(2) << avgSensorsPerCover << "\t"
                << fixed << setprecision(2) << usedMS << "\n";

        // 写调度方案到文件
        {
            string schedFile = "schedule_" + to_string(N) + "_" + to_string(RR) + ".txt";
            ofstream fs(schedFile);
            if (fs.is_open()) {
                fs << "调度结果(每次覆盖1时间单位):\n";
                for (size_t i = 0; i < result.schedule.size(); i++) {
                    fs << (i + 1) 
                       << ", (CoverageSetID=" << result.schedule[i] << ")\n";
                }
                fs << "总共使用了 " << result.schedule.size()
                   << " 次覆盖集, 生命期=" << result.lifetime << "\n";
                fs.close();
            }
        }

        cout << "用时: " << usedMS << " ms\n\n";
    }

    foutSum.close();
    return 0;
}
