package com.streamnova.stream;

import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Service;
import java.time.Instant;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

@Service
public class StreamScheduler {
    private final Map<String, Instant> scheduledStreams = new ConcurrentHashMap<>();

    public void scheduleStream(String streamId, Instant startTime) {
        scheduledStreams.put(streamId, startTime);
    }

    @Scheduled(fixedRate = 60000)
    public void checkAndStartScheduledStreams() {
        Instant now = Instant.now();
        scheduledStreams.forEach((id, time) -> {
            if (time.isBefore(now)) {
                startBroadcast(id);
                scheduledStreams.remove(id);
            }
        });
    }

    private void startBroadcast(String streamId) {
        System.out.println("Activating scheduled live broadcast for stream: " + streamId);
    }
}
