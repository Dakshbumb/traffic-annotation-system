from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, Text, ForeignKey, DateTime, JSON
from sqlalchemy.orm import relationship

from database import Base


class Video(Base):
    __tablename__ = "videos"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, unique=True, index=True)        # stored filename
    original_filename = Column(String)                        # uploaded name
    created_at = Column(DateTime, default=datetime.utcnow)

    annotations = relationship(
        "Annotation",
        back_populates="video",
        cascade="all, delete-orphan",
    )


class Annotation(Base):
    __tablename__ = "annotations"

    id = Column(Integer, primary_key=True, index=True)
    video_id = Column(Integer, ForeignKey("videos.id"), index=True)
    frame_index = Column(Integer, index=True)
    track_id = Column(Integer, nullable=True)                # DeepSORT track ID (optional)
    class_label = Column(String, index=True)                 # car, bike, pedestrian, signal, lane
    x1 = Column(Float)
    y1 = Column(Float)
    x2 = Column(Float)
    y2 = Column(Float)
    extra_meta = Column(JSON, nullable=True)                  # stored as real JSON/dict
# JSON string (speed, violations, etc.)
    created_at = Column(DateTime, default=datetime.utcnow)

    video = relationship("Video", back_populates="annotations")

class AutolabelJob(Base):
    __tablename__ = "autolabel_jobs"

    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(String, unique=True, index=True, nullable=False)
    video_id = Column(Integer, ForeignKey("videos.id"), nullable=False)
    status = Column(String, nullable=False, default="queued")  # queued | running | completed | failed
    progress = Column(Float, nullable=False, default=0.0)      # 0.0 – 1.0
    params = Column(JSON, nullable=True)                       # store request params
    error_message = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)

    video = relationship("Video")


class AnalyticsLine(Base):
    __tablename__ = "analytics_lines"

    id = Column(Integer, primary_key=True, index=True)
    video_id = Column(Integer, ForeignKey("videos.id"), index=True)
    name = Column(String)                             # e.g., "Main Road Counting"
    line_type = Column(String)                        # "count" or "calibration"
    points = Column(JSON)                             # [[x1,y1], [x2,y2]]
    meta_data = Column(JSON, nullable=True)           # e.g. {"distance_meters": 20.0} or {"count_in": 10, "count_out": 5}
    created_at = Column(DateTime, default=datetime.utcnow)

    video = relationship("Video", back_populates="analytics_lines")


class LaneZone(Base):
    """Defines a lane zone as a polygon for cut-in/cut-out detection."""
    __tablename__ = "lane_zones"

    id = Column(Integer, primary_key=True, index=True)
    video_id = Column(Integer, ForeignKey("videos.id"), index=True)
    name = Column(String)                             # e.g., "Ego Lane", "Left Lane"
    zone_type = Column(String)                        # "ego", "adjacent_left", "adjacent_right"
    points = Column(JSON)                             # [[x1,y1], [x2,y2], [x3,y3], ...]  polygon vertices
    color = Column(String, default="#22c55e")         # Display color
    created_at = Column(DateTime, default=datetime.utcnow)

    video = relationship("Video", back_populates="lane_zones")


class LaneEvent(Base):
    """Stores detected cut-in and cut-out events."""
    __tablename__ = "lane_events"

    id = Column(Integer, primary_key=True, index=True)
    video_id = Column(Integer, ForeignKey("videos.id"), index=True)
    track_id = Column(Integer, index=True)            # Vehicle track ID from DeepSORT
    event_type = Column(String, index=True)           # "cut_in" or "cut_out"
    frame_index = Column(Integer, index=True)         # Frame when event occurred
    from_zone = Column(String)                        # Zone vehicle came from
    to_zone = Column(String)                          # Zone vehicle moved to
    confidence = Column(Float, default=1.0)           # Detection confidence
    bbox = Column(JSON)                               # Bounding box at event: [x1, y1, x2, y2]
    created_at = Column(DateTime, default=datetime.utcnow)

    video = relationship("Video", back_populates="lane_events")


class NearMissEvent(Base):
    """Stores detected near-miss/collision risk events between two vehicles."""
    __tablename__ = "near_miss_events"

    id = Column(Integer, primary_key=True, index=True)
    video_id = Column(Integer, ForeignKey("videos.id"), index=True)
    frame_index = Column(Integer, index=True)
    track_id_1 = Column(Integer)
    track_id_2 = Column(Integer)
    ttc: float = Column(Float)            # Time To Collision in seconds
    distance: float = Column(Float)       # Pixel distance at event
    relative_speed: float = Column(Float) # Relative speed in pixels/frame (or calibrated units)
    severity = Column(String)             # "info", "warning", "critical"
    created_at = Column(DateTime, default=datetime.utcnow)

    video = relationship("Video", back_populates="near_miss_events")


# Add back-references to Video
Video.analytics_lines = relationship("AnalyticsLine", back_populates="video", cascade="all, delete-orphan")
Video.lane_zones = relationship("LaneZone", back_populates="video", cascade="all, delete-orphan")
Video.lane_events = relationship("LaneEvent", back_populates="video", cascade="all, delete-orphan")
Video.near_miss_events = relationship("NearMissEvent", back_populates="video", cascade="all, delete-orphan")

