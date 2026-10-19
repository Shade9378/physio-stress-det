package org.physio.wear

fun canConnectHistory(first: Estimate, second: Estimate): Boolean =
    first.sessionId != 0L && first.sessionId == second.sessionId &&
        first.mode == second.mode && first.modelId == second.modelId &&
        first.modelId != "legacy/unknown"
