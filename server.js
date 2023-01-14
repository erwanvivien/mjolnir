const fs = require("fs");
const path = require("path");

const express = require("express");
const app = express();
const port = 3000;

const http = require("http");
const server = http.createServer(app);

const { Server } = require("socket.io");
const io = new Server(server);

let last_refresh = new Date(0);
const file = path.join(__dirname, "pkg", "mjolnir_bg.wasm");

io.on("connection", (socket) => {
  //   console.log("a user connected");
  socket.on("disconnect", () => {
    // console.log("user disconnected");
  });

  socket.on("refresh", () => {
    // Check if file is newer than last refresh
    // If so, send "update" event to client
    // Else, do nothing

    const now = fs.statSync(file).mtime;
    if (now > last_refresh) {
      last_refresh = now;
      console.log("UPDATE");
      socket.emit("update", now);
    }
  });
});

app.use("/", express.static(__dirname));

server.listen(port, () => {
  console.log(`Example app listening on port ${port}`);
});
