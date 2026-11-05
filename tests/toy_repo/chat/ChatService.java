package com.streamnova.chat;

import org.springframework.messaging.handler.annotation.MessageMapping;
import org.springframework.messaging.handler.annotation.SendTo;
import org.springframework.stereotype.Controller;

@Controller
public class ChatService {
    @MessageMapping("/chat.sendMessage")
    @SendTo("/topic/public")
    public String broadcastChatMessage(String message) {
        return "Broadcasting: " + message;
    }
}
