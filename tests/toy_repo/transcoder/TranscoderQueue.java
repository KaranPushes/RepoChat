package com.streamnova.transcoder;

import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;

public class TranscoderQueue {
    private final BlockingQueue<String> queue = new LinkedBlockingQueue<>(100);

    public boolean submitVideo(String videoId) {
        return queue.offer(videoId);
    }

    public String getNextVideo() throws InterruptedException {
        return queue.take();
    }
}
