.pragma library

// Presentation precision is independent of the value accepted by an editor.
// `decimals` in a descriptor is a storage/validation contract, not a display rule.
function policy(descriptor) {
    let item = descriptor || {}
    let integer = item.decimals === 0 || item.integer === true
    if (item.displayDecimals !== undefined)
        return { decimals: Math.max(0, Math.min(15, Number(item.displayDecimals))), digits: 5, integer: integer }
    if (item.displayDigits !== undefined)
        return { decimals: null, digits: Math.max(1, Math.min(15, Number(item.displayDigits))), integer: integer }
    if (integer)
        return { decimals: 0, digits: 5, integer: true }

    let key = String(item.key || "").toLowerCase()
    let label = String(item.label || item.accessibleName || "").toLowerCase()
    let words = key + " " + label
    let decimals = null
    let signalScale = /intensity|power|background|amplitude|height guess|correction|coefficient|multiplier/.test(label)
    // Explicit units take precedence over generic names such as center/width.
    if (/mev/.test(label) || /(?:min|max|neighbor|noise)?_?(?:width|distance|shift|tolerance)_mev/.test(key))
        decimals = 3
    else if (/\bnm\b|wavelength|temperature|\(k\)|\bkelvin\b/.test(label)
             || /temperature|thetat?min|thetat?max/.test(key))
        decimals = 2
    else if (/angle|\bdeg(?:rees)?\b/.test(label) || /^(?:ink|inkp|outk|outkp|angleoffset|anglecursor|fitmin|fitmax)$/.test(key))
        decimals = /scale|coefficient/.test(words) ? null : 2
    else if (/voltage|magnetic|\(t\)|\(v\)|\btesla\b/.test(label)
             || /^(?:bcursor|max_delta_b|zero_window_t|fieldhalfwidth|fieldmin|fieldmax|thetahalfwidth)$/.test(key))
        decimals = /tolerance|window|delta/.test(words) ? 4 : 3
    else if (/\bev\b/.test(label) || /energy/.test(label)
             || (!signalScale && (/^(?:xmin|xmax|x_min|x_max|splitx|energymin|energymax|featurelow|featurehigh|regionlow|regionhigh|manualreference|manualcenter)$/.test(key)
             || /(?:^|\.)(?:x_?min|x_?max|bandmin|bandmax|boundary\.x[01])$/.test(key)
             || /^bn\..*\.(?:low|high)$|^bn\.taper_width$/.test(key))))
        decimals = /center|manualreference|manualcenter/.test(words) || /^(?:refit\.)?peak\.[0-9]+\.(?:lower|upper)$/.test(key) ? 6 : 4
    else if (/^(?:center|manualcenter|manualreference)$/.test(key) || /(?:^|\.)peak\.[0-9]+\.center$/.test(key))
        decimals = 6
    else if (/^(?:export\.pngwidth|pngwidthscale)$/.test(key))
        decimals = 2
    else if (/^(?:gate|ymin|ymax|y_min|y_max)$/.test(key) && !/power|intensity|count/.test(label))
        decimals = 3
    return { decimals: decimals, digits: 5, integer: false }
}

function trimZeros(text) {
    let parts = text.toLowerCase().split("e")
    if (parts[0].indexOf(".") >= 0)
        parts[0] = parts[0].replace(/0+$/, "").replace(/\.$/, "")
    if (parts.length > 1)
        return parts[0] + "e" + String(Number(parts[1]))
    return parts[0] === "-0" ? "0" : parts[0]
}

function display(value, descriptor) {
    let number = Number(value)
    if (!isFinite(number)) return ""
    if (number === 0) return "0"
    let format = policy(descriptor)
    let magnitude = Math.abs(number)
    // Keep small scientific values visible even for a normally fixed-decimal unit.
    if (magnitude < 1e-3 || magnitude >= 1e6)
        return trimZeros(number.toExponential(format.digits - 1))
    if (format.decimals !== null) {
        let fixed = number.toFixed(format.decimals)
        if (Number(fixed) !== 0)
            return trimZeros(fixed)
    }
    return trimZeros(Number(number.toPrecision(format.digits)).toString())
}

function exact(value) {
    let number = Number(value)
    return isFinite(number) ? (number === 0 ? "0" : String(number)) : ""
}

function parse(text) {
    let source = String(text).trim()
    if (!/^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$/.test(source))
        return undefined
    let number = Number(source)
    let mantissa = source.split(/[eE]/)[0]
    if (number === 0 && /[1-9]/.test(mantissa)) return undefined
    return isFinite(number) ? number : undefined
}
