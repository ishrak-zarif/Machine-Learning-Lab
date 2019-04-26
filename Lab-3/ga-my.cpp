#include <bits/stdc++.h>

using namespace std;

int population=4;
int numberOfBits=6;
vector<pair<vector<int>, int> > g;

int f(int x){
    return -(x*x*x*x)+9;
}

bool cmp(pair<vector<int>, int> a, pair<vector<int>, int> b){
    return a.second>b.second;
}

void printVector(){
    for(int i=0; i<population; i++){
        int sum=0;
        int power=numberOfBits-2;
        printf("%d ",g[i].first[0]);
        for(int j=1; j<numberOfBits; j++){
            printf("%d ",g[i].first[j]);
            sum+=g[i].first[j]*(1<<power);
            power--;
        }
        if(g[i].first[0]==1)
            sum=-sum;
        printf("\t%d \t%d\n", sum, g[i].second);
    }
    printf("---------------------------------\n");
}

int main(){
    srand(time(NULL));
    for(int i=0; i<population; i++){
        vector<int> v;
        for(int j=0; j<numberOfBits; j++){
            v.push_back(rand()%2);
        }
        int power=numberOfBits-2;
        int sum=0;
        for(int j=1; j<numberOfBits; j++){
            sum+=v[j]*(1<<power);
            power--;
        }
        if(v[0]==1)
            sum=-sum;
        g.push_back({v, f(sum)});
    }
    sort(g.begin(), g.end(), cmp);
    int best=g[0].second;

    for(int it=1; it<=100; it++){
        int cr=rand()%numberOfBits;
        if(cr==0)
            cr++;
        for(int i=cr; i<numberOfBits; i++){
            swap(g[0].first[i], g[1].first[i]);
        }
        int rr=rand()%population;
        int ri=rand()%numberOfBits;
        if(ri==0)
            ri++;
        g[rr].first[ri]=1-g[rr].first[ri];
        for(int i=0; i<population; i++){
            int sum=0;
            int power=numberOfBits-2;
            for(int j=1; j<numberOfBits; j++){
                    sum+=g[i].first[j]*(1<<power);
                    power--;
            }
            if(g[i].first[0] == 1)
                sum = -sum;
            g[i].second = f(sum);
        }
        sort(g.begin(), g.end(), cmp);
        best = max(best, g[0].second);
        printf("Iteration %d:\n", it);
        printVector();
    }
    cout << "Best value: " << best << endl;

    return 0;
}
