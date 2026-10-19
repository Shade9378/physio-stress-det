package org.physio.wear

import android.content.ContentValues
import android.content.Context
import android.database.sqlite.SQLiteDatabase
import android.database.sqlite.SQLiteOpenHelper

data class Estimate(val timestamp: Long, val probability: Float, val mode: String,
                    val latencyMs: Double, val target: Int?, val sessionId: Long = 0,
                    val modelId: String = "legacy/unknown")

class HistoryStore(context: Context, name: String = "stress.db") : SQLiteOpenHelper(context, name, null, 3) {
    override fun onCreate(db: SQLiteDatabase) {
        db.execSQL("CREATE TABLE estimates (timestamp INTEGER, probability REAL, mode TEXT, latency REAL, target INTEGER, session INTEGER NOT NULL DEFAULT 0, model TEXT NOT NULL DEFAULT 'legacy/unknown')")
    }
    override fun onUpgrade(db: SQLiteDatabase, oldVersion: Int, newVersion: Int) {
        if (oldVersion < 2) db.execSQL("ALTER TABLE estimates ADD COLUMN session INTEGER NOT NULL DEFAULT 0")
        if (oldVersion < 3) db.execSQL("ALTER TABLE estimates ADD COLUMN model TEXT NOT NULL DEFAULT 'legacy/unknown'")
    }

    fun insert(value: Estimate) {
        writableDatabase.insertOrThrow("estimates", null, ContentValues().apply {
            put("timestamp", value.timestamp)
            put("probability", value.probability)
            put("mode", value.mode)
            put("latency", value.latencyMs)
            put("session", value.sessionId)
            put("model", value.modelId)
            if (value.target == null) putNull("target") else put("target", value.target)
        })
        // Retain at most 2,000 local summaries; never store raw physiological samples.
        writableDatabase.execSQL("DELETE FROM estimates WHERE rowid NOT IN (SELECT rowid FROM estimates ORDER BY rowid DESC LIMIT 2000)")
    }

    fun recent(): List<Estimate> = readableDatabase.rawQuery(
        "SELECT timestamp, probability, mode, latency, target, session, model FROM estimates ORDER BY rowid DESC LIMIT 120", null
    ).use { cursor ->
        buildList {
            while (cursor.moveToNext()) add(Estimate(cursor.getLong(0), cursor.getFloat(1),
                cursor.getString(2), cursor.getDouble(3), if (cursor.isNull(4)) null else cursor.getInt(4), cursor.getLong(5), cursor.getString(6)))
        }
    }

    fun clear() { writableDatabase.delete("estimates", null, null) }
}
