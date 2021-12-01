# network-trace-application-predictor
CSE 824 Project using MobileInsight as a tool for collecting cellular network traces while accessing internet applications. Goal is to predict the correct application given our network trace data as input.


Notes:

Features to consider:

Frame Size, Number of frames (receiving and sending), frame distribution information
Data rate (receiving and sending)
Frame interarrival time

Need to consider data frames, control frames,
management frames

Algorithms to consider:
SVM and NN

SVM:
- Good for small training sets

NN:
- Able to tolerate noise

Use RBF possibly for both SVM and NN
Or maybe consider RBFN

Will conduct experiments on home network (WPA2 or 3 security)


6 Categories to consider:
Browsing, Chatting, Online Game, Downloading, Uploading, Online Video

Or 4 broader categories
Tools (Browsing, Upload, Mailing)

Social

Entertainment

Shopping


20 Applications we can consider:
1. Google Chrome
2. Mozilla Firefox
3. Facebook
4. Instagram
5. Messenger
6. Snapchat
7. Youtube
8. Netflix
9. Amazon Shopping
10. Tiktok
11. Walmart Shopping
12. Gmail
13. Twitter
14. Roblox
15. Minecraft
16. Pinterest
17. Outlook
18. Google Drive
19. Amazon Prime
20. Reddit

Extra Considerations:
21. Discord
22. Target Shopping
23. Crypto.com
24. Coinbase
25. Best Buy shopping

link for more: https://sensortower.com/ios/rankings/top/iphone/us/all-categories?date=2021-11-27


Labels example input:
[10, 11, 2, 20, 19, 1, 4, 6, 9, etc.] having each number map to an application

Data example features:
[App name, MAC_Subpacket_Size, MAC_Subpacket_Sample_Count, MAC_BSR, MAC_Grant_Received, MAC_Grant_Utilized, etc.]

Example format:
[['Airbnb', 1, 1, 10, 192, 84, 90],
['Walmart', 10, 6, 24, 2, 25, 96],
['Snapchat', 12, 245, 500, 9, 4, 0]]

After encoding:
[[1, 1, 1, 10, 192, 84, 90],
[11, 10, 6, 24, 2, 25, 96],
[6, 12, 245, 500, 9, 4, 0]]