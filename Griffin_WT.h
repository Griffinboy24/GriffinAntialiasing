#pragma once
#include <JuceHeader.h>
#include <vector>
#include <cmath>
#include <cstdint>
#include <cstring>


#include "src/griffinwave4/BaseVoiceState.cpp"
#include "src/griffinwave4/BaseVoiceState.h"
#include "src/griffinwave4/Downsampler2Flt.hpp"
#include "src/griffinwave4/rspl.hpp"
#include "src/griffinwave4/InterpFlt.hpp"
#include "src/griffinwave4/InterpPack.cpp"
#include "src/griffinwave4/InterpPack.h"
#include "src/griffinwave4/MipMapFlt.hpp"
#include "src/griffinwave4/ResamplerFlt.cpp"
#include "src/griffinwave4/ResamplerFlt.h"


#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace project
{
    using namespace juce;
    using namespace hise;
    using namespace scriptnode;

    template<int NV>
    struct Griffin_WT : data::base
    {
        SNEX_NODE(Griffin_WT);
        struct MetadataClass { SN_NODE_ID("Griffin_WT"); };

        static constexpr bool isModNode() { return false; }
        static constexpr bool isPolyphonic() { return NV > 1; }
        static constexpr bool hasTail() { return false; }
        static constexpr bool isSuspendedOnSilence() { return false; }
        static constexpr int  getFixChannelAmount() { return 2; }
        static constexpr int  NumTables = 0;
        static constexpr int  NumSliderPacks = 0;
        static constexpr int  NumAudioFiles = 0;
        static constexpr int  NumFilters = 0;
        static constexpr int  NumDisplayBuffers = 0;

        Griffin_WT() : ready(false), currentOct(0.0) {}

        void prepare(PrepareSpecs /*specs*/)
        {
            static constexpr int padCycles = 4;
            static constexpr int origCycles = 256;
            static constexpr int cycleLen = 2048;
            static constexpr int totalCycles = padCycles + origCycles;
            const int totalLen = totalCycles * cycleLen;

            sampleBuffer.resize(totalLen);
            double sawInc = 2.0 / double(cycleLen);
            int phase = 0;
            for (int i = 0; i < totalLen; ++i)
            {
                sampleBuffer[i] = float(-1.0 + phase * sawInc);
                if (++phase >= cycleLen)
                    phase = 0;
            }

            mipMap.init_sample(
                totalLen,
                rspl::InterpPack::get_len_pre(),
                rspl::InterpPack::get_len_post(),
                12,
                rspl::ResamplerFlt::_fir_mip_map_coef_arr,
                rspl::ResamplerFlt::MIP_MAP_FIR_LEN
            );
            mipMap.fill_sample(sampleBuffer.data(), totalLen);

            resampler.set_interp(interpPack);
            resampler.set_sample(mipMap);
            resampler.clear_buffers();

            wrapOffset = (rspl::Int64(padCycles * cycleLen) << 32);
            wrapMask = ((rspl::Int64(cycleLen) << 32) - 1);

            ready = true;
        }

        void reset() {}

        template<typename PD>
        void process(PD& d)
        {
            if (!ready) return;

            auto& fix = d.template as<ProcessData<2>>();
            auto  blk = fix.toAudioBlock();
            float* L = blk.getChannelPointer(0);
            float* R = blk.getChannelPointer(1);
            int    n = d.getNumSamples();

            // render mono into left buffer
            for (int i = 0; i < n; ++i)
            {
                rspl::Int64 pos = resampler.get_playback_pos();
                pos = (pos & wrapMask) + wrapOffset;
                resampler.set_playback_pos(pos);

                long bits = long(currentOct * (1L << rspl::BaseVoiceState::NBR_BITS_PER_OCT));
                resampler.set_pitch(bits);

                float s = 0.0f;
                resampler.interpolate_block(&s, 1);
                L[i] = s;
            }

            // scale down volume and copy to right channel
            constexpr float gain = 0.9f;
            FloatVectorOperations::multiply(L, L, gain, n);
            FloatVectorOperations::copy(R, L, n);
        }

        template<int P>
        void setParameter(double vOct)
        {
            if constexpr (P == 0)
                currentOct = vOct;
        }

        void createParameters(ParameterDataList& list)
        {
            parameter::data p("Pitch", { 0.0, 9.0, 0.000001 });
            p.setDefaultValue(0.0);
            registerCallback<0>(p);
            list.add(std::move(p));
        }

        SN_EMPTY_PROCESS_FRAME;
        void handleHiseEvent(HiseEvent&) {}

    private:
        bool               ready;
        double             currentOct;
        rspl::InterpPack   interpPack;
        rspl::MipMapFlt    mipMap;
        rspl::ResamplerFlt resampler;
        std::vector<float> sampleBuffer;
        rspl::Int64        wrapOffset;
        rspl::Int64        wrapMask;
    };
}
