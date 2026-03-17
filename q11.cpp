#include <iostream>
#include <vector>
#include <chrono>
#include <cstdlib>
using namespace std;

int findSum(vector<int> &arr){
    int sum = 0;
    for(int i=0;i<arr.size();i++)
        sum += arr[i];
    return sum;
}

int searchKey(vector<int> &arr, int key){
    for(int i=0;i<arr.size();i++){
        if(arr[i] == key)
            return i;
    }
    return -1;
}

int main(){
    int n;
    cout<<"Enter array size: ";
    cin>>n;

    vector<int> arr(n);

    for(int i=0;i<n;i++)
        arr[i] = rand()%100;

    int key;
    cout<<"Enter key to search: ";
    cin>>key;

    auto start = chrono::high_resolution_clock::now();

    int sum = findSum(arr);
    int pos = searchKey(arr,key);

    auto end = chrono::high_resolution_clock::now();

    chrono::duration<double> time = end-start;

    cout<<"Sum = "<<sum<<endl;

    if(pos!=-1)
        cout<<"Key found at index "<<pos<<endl;
    else
        cout<<"Key not found"<<endl;

    cout<<"Execution Time = "<<time.count()<<" seconds";

    return 0;
}