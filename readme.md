TO run node server automatically follow below steps

1. npm install -g pm2    

2. pm2 start server.js --name "proctorsense"        

3. To list : pm2 list    

4. To Stop: pm2 stop proctorsense      

5. To start : pm2 start proctorsense      


6. Create a new vd (npserver) and point it to folder containing server.js

IF Faced with error in creation in server do the following
1. where pm2
Fix PATH permanently : setx PATH "%PATH%;C:\Users\<username>\AppData\Roaming\npm"