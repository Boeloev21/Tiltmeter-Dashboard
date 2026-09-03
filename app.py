import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dateutil.relativedelta import relativedelta

try:
    import pandera.pandas as pa
except ImportError:  # older pandera versions expose the API at top level
    import pandera as pa


# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

# Selector options -> relativedelta used to compute the lookback window for
# the "moving average max" annotation.
MA_MAX_TIMEFRAME_DELTAS = {
    "Last Year": relativedelta(years=1),
    "Last 6 Months": relativedelta(months=6),
    "Last 3 Months": relativedelta(months=3),
    "Last Month": relativedelta(months=1),
}
MA_MAX_TIMEFRAME_OPTIONS = list(MA_MAX_TIMEFRAME_DELTAS.keys()) + ["None"]


def plot_tilt_trend(
    df: pd.DataFrame,
    title: str,
    x_col: str = "SensorDateUTC",
    y_col: str = "X_Movement",
    y_units: str = "mm",
    window_days: int = 21,
    show_raw: bool = True,
    show_ma: bool = True,
    ma_max_timeframe: str | None = None,
) -> go.Figure:
    """Line chart of a tiltmeter movement trend (X or Y direction) with
    min/max annotations and a rolling N-day average line.

    show_raw / show_ma toggle whether each series is drawn at all.
    ma_max_timeframe, if given (one of MA_MAX_TIMEFRAME_DELTAS' keys),
    adds an annotation marking the maximum of the moving-average series
    within that trailing window (measured back from the last date in df).
    """

    df = df.sort_values(x_col).reset_index(drop=True)

    min_val = df[y_col].min()
    max_val = df[y_col].max()
    min_time = df[x_col][df[y_col].idxmin()]
    max_time = df[x_col][df[y_col].idxmax()]

    span_days = (df[x_col].max() - df[x_col].min()).total_seconds() / 86400

    rolling_col = None
    if span_days > 0 and len(df) > 1:
        actual_readings_per_day = len(df) / span_days
        full_window = actual_readings_per_day * window_days
        min_periods = max(1, int(full_window * 0.7))

        rolling_col = f"{y_col}_{window_days}D_mean"
        df_indexed = df.set_index(x_col)
        df[rolling_col] = (
            df_indexed[y_col].rolling(f"{window_days}D", min_periods=min_periods).mean().values
        )

    fig = go.Figure()
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title=f"Movement ({y_units})",
    )

    if show_raw:
        fig.add_trace(
            go.Scatter(
                x=df[x_col],
                y=df[y_col],
                mode="lines",
                name=y_col,
            )
        )

        fig.add_trace(
            go.Scatter(
                x=[min_time],
                y=[min_val],
                mode="markers+text",
                marker=dict(color="red", size=10),
                text=[f"Min: {min_val:.3f}{y_units} at {min_time.strftime('%Y-%m-%d %H:%M:%S')}"],
                textposition="bottom center",
                name="Minimum",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[max_time],
                y=[max_val],
                mode="markers+text",
                marker=dict(color="green", size=10),
                text=[f"Max: {max_val:.3f}{y_units} at {max_time.strftime('%Y-%m-%d %H:%M:%S')}"],
                textposition="top center",
                name="Maximum",
            )
        )

    if rolling_col is not None and show_ma:
        fig.add_trace(
            go.Scatter(
                x=df[x_col],
                y=df[rolling_col],
                mode="lines",
                line=dict(color="orange", width=2),
                name=f"{window_days} Day Moving Average",
            )
        )

        if ma_max_timeframe and ma_max_timeframe in MA_MAX_TIMEFRAME_DELTAS:
            window_start = df[x_col].max() - MA_MAX_TIMEFRAME_DELTAS[ma_max_timeframe]
            ma_window_df = df.loc[df[x_col] >= window_start, [x_col, rolling_col]].dropna()

            if not ma_window_df.empty:
                ma_max_idx = ma_window_df[rolling_col].idxmax()
                ma_max_val = ma_window_df.loc[ma_max_idx, rolling_col]
                ma_max_time = ma_window_df.loc[ma_max_idx, x_col]

                fig.add_trace(
                    go.Scatter(
                        x=[ma_max_time],
                        y=[ma_max_val],
                        mode="markers+text",
                        marker=dict(color="purple", size=12, symbol="diamond"),
                        text=[
                            f"MA Max ({ma_max_timeframe}): {ma_max_val:.3f}{y_units} "
                            f"at {ma_max_time.strftime('%Y-%m-%d')}"
                        ],
                        textposition="top center",
                        name=f"MA Max ({ma_max_timeframe})",
                    )
                )

    return fig


def validate_tilt(
    df: pd.DataFrame,
    lower_bound: float,
    upper_bound: float,
    y_col: str,
    nullable: bool = False,
) -> pd.DataFrame:
    """Validate a tilt movement column (X or Y direction) against bounds.
    Raises pa.errors.SchemaErrors (lazily collected) if any row fails."""

    schema = pa.DataFrameSchema(
        {
            y_col: pa.Column(
                dtype=float,
                checks=[
                    pa.Check.between(lower_bound, upper_bound, include_min=True, include_max=True),
                    pa.Check(lambda x: x.notna(), element_wise=False, error=f"{y_col} contains null values"),
                ],
                nullable=nullable,
                coerce=True,
                description=f"{y_col} values must be between {lower_bound} and {upper_bound}",
            )
        }
    )
    validated_df = schema.validate(df, lazy=True)
    return validated_df


def clean_tilt(
    df: pd.DataFrame,
    lower_bound: float,
    upper_bound: float,
    y_col: str,
    nullable: bool = False,
):
    """Remove rows that fail validation for a given movement column.
    Returns (cleaned_df, failure_cases_df_or_None)."""
    try:
        validated_df = validate_tilt(df, lower_bound, upper_bound, y_col, nullable)
        return validated_df, None
    except pa.errors.SchemaErrors as e:
        invalid_indices = e.failure_cases["index"].dropna().unique()
        cleaned_df = df.drop(index=invalid_indices)
        return cleaned_df, e.failure_cases


def get_monthly_readings(df, x_col="SensorDateUTC", y_col="X_Movement", n_months=None):
    """Reading closest to (at or before) the start date, then the same
    calendar date each successive month, via as-of lookup."""

    df = df.sort_values(x_col).reset_index(drop=True)
    s = df.set_index(x_col)[y_col]

    start_date = df[x_col].min()
    end_date = df[x_col].max()

    if n_months is None:
        n_months = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month) + 1

    records = []
    for i in range(n_months):
        target = start_date + relativedelta(months=i)
        if target > end_date:
            break
        value = s.asof(target)
        actual_date = s.index[s.index <= target].max()
        records.append({"target_date": target, "actual_date": actual_date, y_col: value})

    return pd.DataFrame(records)


def get_readings_at_dates(df, target_dates, x_col="SensorDateUTC", y_col="X_Movement", tolerance_days=None):
    """Look up the reading closest to (at or before) each date in
    target_dates via as-of lookup. Optionally enforce a max gap
    (tolerance_days) beyond which the result is NaN."""

    df = df.sort_values(x_col).reset_index(drop=True)
    s = df.set_index(x_col)[y_col]

    records = []
    for target in target_dates:
        target = pd.Timestamp(target)

        # Align target's tz-awareness with the index's, so asof() can compare them
        if s.index.tz is not None:
            if target.tz is None:
                target = target.tz_localize(s.index.tz)
            else:
                target = target.tz_convert(s.index.tz)
        elif target.tz is not None:
            target = target.tz_localize(None)

        value = s.asof(target)
        prior = s.index[s.index <= target]
        actual_date = prior.max() if len(prior) else pd.NaT
        gap_days = (target - actual_date).total_seconds() / 86400 if pd.notna(actual_date) else None

        if tolerance_days is not None and gap_days is not None and gap_days > tolerance_days:
            value = None

        records.append(
            {
                "target_date": target,
                "actual_date": actual_date,
                "gap_days": round(gap_days, 2) if gap_days is not None else None,
                y_col: value,
            }
        )

    return pd.DataFrame(records)


def render_target_section(
    df: pd.DataFrame,
    sensor_name: str,
    axis_label: str,
    x_col: str,
    y_col: str,
    y_units: str,
    window_days: int,
    show_raw: bool,
    show_ma: bool,
    ma_max_timeframe: str | None,
    lower_bound: float,
    upper_bound: float,
    nullable: bool,
    key_prefix: str,
):
    """Render the full analysis section (raw plot, validation/cleaning,
    cleaned plot, monthly readings, custom date lookup) for a single
    movement column (X or Y direction). Assumes df already has x_col
    parsed as datetime and dropna'd."""

    if not show_raw and not show_ma:
        st.warning("Both series are hidden — toggle at least one on in the sidebar to see a chart.")

    # --- Raw plot ---
    st.subheader(f"Raw trend — {axis_label}")
    fig_raw = plot_tilt_trend(
        df, title=f"{sensor_name} {axis_label} Trend with Min/Max Annotations",
        x_col=x_col, y_col=y_col, y_units=y_units, window_days=window_days,
        show_raw=show_raw, show_ma=show_ma, ma_max_timeframe=ma_max_timeframe,
    )
    st.plotly_chart(fig_raw, use_container_width=True, key=f"raw_{key_prefix}")

    # --- Validate & clean ---
    st.subheader(f"Validation & cleaning — {axis_label}")
    cleaned_df, failure_cases = clean_tilt(df, lower_bound, upper_bound, y_col, nullable)

    if failure_cases is None:
        st.success("Validation passed — no cleaning required.")
    else:
        st.warning(f"Removed {len(failure_cases['index'].dropna().unique())} invalid row(s).")
        with st.expander("View failed rows"):
            st.dataframe(failure_cases, use_container_width=True)

    with st.expander("Cleaned data summary (describe())"):
        st.dataframe(cleaned_df[[y_col]].describe(), use_container_width=True)

    # --- Cleaned plot ---
    st.subheader(f"Cleaned trend — {axis_label}")
    fig_clean = plot_tilt_trend(
        cleaned_df, title=f"{sensor_name} {axis_label} Cleaned Trend with Min/Max Annotations",
        x_col=x_col, y_col=y_col, y_units=y_units, window_days=window_days,
        show_raw=show_raw, show_ma=show_ma, ma_max_timeframe=ma_max_timeframe,
    )
    st.plotly_chart(fig_clean, use_container_width=True, key=f"clean_{key_prefix}")

    st.download_button(
        f"⬇️ Download cleaned {axis_label} data (CSV)",
        data=cleaned_df.to_csv(index=False).encode("utf-8"),
        file_name=f"{sensor_name}_{y_col}_cleaned.csv",
        mime="text/csv",
        key=f"dl_clean_{key_prefix}",
    )

    # --- Monthly readings ---
    st.subheader(f"Monthly readings — {axis_label}")
    monthly_df = get_monthly_readings(cleaned_df, x_col=x_col, y_col=y_col)
    st.dataframe(monthly_df, use_container_width=True)
    st.download_button(
        f"⬇️ Download monthly {axis_label} readings (CSV)",
        data=monthly_df.to_csv(index=False).encode("utf-8"),
        file_name=f"{sensor_name}_{y_col}_monthly_readings.csv",
        mime="text/csv",
        key=f"dl_monthly_{key_prefix}",
    )

    # --- Custom date lookup ---
    st.subheader(f"Reading at specific dates — {axis_label}")
    st.caption(
        "Pick any dates (e.g. the three dates for a report) to get the reading at or "
        "immediately before each one."
    )

    default_dates = [
        cleaned_df[x_col].min().date(),
        cleaned_df[x_col].max().date(),
    ]
    picked_dates = st.date_input(
        "Target date(s)",
        value=default_dates,
        key=f"dates_{key_prefix}",
    )
    # st.date_input returns a single date or a tuple depending on selection
    if isinstance(picked_dates, (list, tuple)):
        date_list = list(picked_dates)
    else:
        date_list = [picked_dates]

    tolerance = st.number_input(
        "Max acceptable gap (days) — leave 0 to disable",
        min_value=0, value=0, key=f"tol_{key_prefix}",
    )

    if date_list:
        lookup_df = get_readings_at_dates(
            cleaned_df, date_list, x_col=x_col, y_col=y_col,
            tolerance_days=tolerance if tolerance > 0 else None,
        )
        st.dataframe(lookup_df, use_container_width=True)
        st.download_button(
            f"⬇️ Download {axis_label} date lookup (CSV)",
            data=lookup_df.to_csv(index=False).encode("utf-8"),
            file_name=f"{sensor_name}_{y_col}_date_lookup.csv",
            mime="text/csv",
            key=f"dl_lookup_{key_prefix}",
        )


# --------------------------------------------------------------------------
# Streamlit app
# --------------------------------------------------------------------------

st.set_page_config(page_title="Tiltmeter Dashboard", layout="wide")
st.title("📐 Tiltmeter Movement Dashboard")
st.caption(
    "Upload one or more Tiltmeter sensor CSVs to plot X/Y movement trends, "
    "validate & clean the data, and pull readings at monthly or custom dates."
)

# ---- Sidebar: global settings ----
with st.sidebar:
    st.header("Settings")

    uploaded_files = st.file_uploader(
        "Upload Tiltmeter CSV file(s)",
        type=["csv"],
        accept_multiple_files=True,
        help="You can select multiple files at once (e.g. TILT01.csv, TILT02.csv, ...).",
    )

    st.subheader("Column mapping")
    x_col = st.text_input("Date column", value="SensorDateUTC")
    x_movement_col = st.text_input("X direction movement column", value="X_Movement")
    y_movement_col = st.text_input("Y direction movement column", value="Y_Movement")
    y_units = st.text_input("Units", value="mm")

    st.subheader("Rolling average")
    window_days = st.number_input("Window (days)", min_value=1, max_value=365, value=21)

    st.subheader("Chart display")
    show_raw = st.checkbox("Show raw data series", value=True)
    show_ma = st.checkbox("Show moving average series", value=True)
    ma_max_timeframe = st.selectbox(
        "Highlight moving-average max over",
        MA_MAX_TIMEFRAME_OPTIONS,
        index=0,
        help="Adds a marker at the peak of the moving-average line within this trailing window.",
    )
    if ma_max_timeframe == "None":
        ma_max_timeframe = None

    st.subheader("Validation bounds")
    st.caption("Set independently for each movement direction.")
    bcol_x, bcol_y = st.columns(2)
    with bcol_x:
        st.markdown("**X direction**")
        x_lower_bound = st.number_input("X lower bound", value=-50.0, step=1.0, key="x_lower")
        x_upper_bound = st.number_input("X upper bound", value=50.0, step=1.0, key="x_upper")
    with bcol_y:
        st.markdown("**Y direction**")
        y_lower_bound = st.number_input("Y lower bound", value=-50.0, step=1.0, key="y_lower")
        y_upper_bound = st.number_input("Y upper bound", value=50.0, step=1.0, key="y_upper")
    nullable = st.checkbox("Allow null values", value=False)

    st.caption(
        "Tip: if a particular sensor is known to have a wider/narrower expected "
        "range of movement, adjust the bounds above before reviewing its tab."
    )

if not uploaded_files:
    st.info("👆 Upload one or more CSV files in the sidebar to get started.")
    st.stop()

# ---- One tab per uploaded file ----
tab_labels = [f.name for f in uploaded_files]
tabs = st.tabs(tab_labels)

for file, tab in zip(uploaded_files, tabs):
    with tab:
        try:
            sensor_name = file.name.rsplit(".", 1)[0]

            # --- Load ---
            try:
                raw_df = pd.read_csv(file)
            except Exception as e:
                st.error(f"Could not read {file.name}: {e}")
                continue

            required_cols = [x_col, x_movement_col, y_movement_col]
            missing_cols = [c for c in required_cols if c not in raw_df.columns]
            if missing_cols:
                st.error(
                    f"Expected column(s) {missing_cols} not found in {file.name}. "
                    f"Available columns: {list(raw_df.columns)}"
                )
                continue

            df = raw_df.copy()
            df[x_col] = pd.to_datetime(df[x_col], errors="coerce")
            n_unparsed = df[x_col].isna().sum()
            df = df.dropna(subset=[x_col])

            col1, col2, col3 = st.columns(3)
            col1.metric("Rows", len(df))
            col2.metric("Date range (days)", f"{(df[x_col].max() - df[x_col].min()).days}")
            col3.metric("Unparsable dates dropped", n_unparsed)

            # ---- One inner tab per movement direction ----
            x_tab, y_tab = st.tabs(["↔️ X Direction", "↕️ Y Direction"])

            with x_tab:
                render_target_section(
                    df, sensor_name=sensor_name, axis_label="X Direction",
                    x_col=x_col, y_col=x_movement_col, y_units=y_units,
                    window_days=window_days, show_raw=show_raw, show_ma=show_ma,
                    ma_max_timeframe=ma_max_timeframe,
                    lower_bound=x_lower_bound, upper_bound=x_upper_bound, nullable=nullable,
                    key_prefix=f"{sensor_name}_x",
                )

            with y_tab:
                render_target_section(
                    df, sensor_name=sensor_name, axis_label="Y Direction",
                    x_col=x_col, y_col=y_movement_col, y_units=y_units,
                    window_days=window_days, show_raw=show_raw, show_ma=show_ma,
                    ma_max_timeframe=ma_max_timeframe,
                    lower_bound=y_lower_bound, upper_bound=y_upper_bound, nullable=nullable,
                    key_prefix=f"{sensor_name}_y",
                )

        except Exception as e:
            st.error(f"Unexpected error processing {file.name}: {e}")
