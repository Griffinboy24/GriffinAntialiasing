#pragma once
#include <JuceHeader.h>
#include <cstring>
#include <vector>
#include <cmath>

/*
// Backup Don't remove
#include "src/griffinwave4/BaseVoiceState.cpp"
#include "src/griffinwave4/BaseVoiceState.h"
#include "src/griffinwave4/Downsampler2Flt.cpp"
#include "src/griffinwave4/Downsampler2Flt.h"
#include "src/griffinwave4/Downsampler2Flt.hpp"
#include "src/griffinwave4/rspl.hpp"
#include "src/griffinwave4/InterpFlt.h"
#include "src/griffinwave4/InterpFlt.hpp"
#include "src/griffinwave4/InterpFltPhase.h"
#include "src/griffinwave4/InterpFltPhase.hpp"
#include "src/griffinwave4/InterpPack.cpp"
#include "src/griffinwave4/InterpPack.h"
#include "src/griffinwave4/MipMapFlt.cpp"
#include "src/griffinwave4/MipMapFlt.h"
#include "src/griffinwave4/MipMapFlt.hpp"
#include "src/griffinwave4/ResamplerFlt.cpp"
#include "src/griffinwave4/ResamplerFlt.h"
*/

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

namespace project
{

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

    using namespace juce;
    using namespace hise;
    using namespace scriptnode;

    template <int NV>
    struct Griffin_WT : public data::base
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

        Griffin_WT() {}

        void prepare(PrepareSpecs specs)
        {
            sampleRate = specs.sampleRate;
            if (!testsGenerated)
            {
                generateTests();
                testsGenerated = true;
            }
        }

        void reset()
        {
            playPosition = 0;
            playToggle = false;
        }

        template <typename ProcessDataType>
        void process(ProcessDataType& data)
        {
            auto& fixData = data.template as<ProcessData<getFixChannelAmount()>>();
            auto  audioBlock = fixData.toAudioBlock();
            float* left = audioBlock.getChannelPointer(0);
            float* right = audioBlock.getChannelPointer(1);
            int    n = data.getNumSamples();

            if (playToggle)
            {
                for (int i = 0; i < n; ++i)
                {
                    float s = 0.0f;
                    if (playPosition < testBuffer.size())
                        s = testBuffer[playPosition++];
                    left[i] = s;
                    right[i] = s;
                }
            }
            else
            {
                for (int i = 0; i < n; ++i)
                {
                    left[i] = 0.0f;
                    right[i] = 0.0f;
                }
            }
        }

    private:
        double             sampleRate{ 44100.0 };
        std::vector<float> testBuffer;
        uint64_t           playPosition{ 0 };
        bool               playToggle{ false };
        bool               testsGenerated{ false };

        // copy from main.cpp
        static void generate_steady_sine(std::vector<float>& v, long len, double freq)
        {
            v.resize(len);
            for (long i = 0; i < len; ++i)
                v[i] = std::cos(i * freq * (2 * M_PI));
        }

        static void generate_steady_saw(std::vector<float>& v, long len, long wavelength)
        {
            v.resize(len);
            double val = 0.0;
            double step = 2.0 / double(wavelength - 1);
            for (long i = 0; i < len; ++i)
            {
                if ((i % wavelength) == 0) val = -1.0;
                v[i] = float(val);
                val += step;
            }
        }

        void generateTests()
        {
            using namespace rspl;
            const double fs = sampleRate;

            // --- SINE SWEEP (10s), map level = 12 ------------------------
            const double  fc1 = 15000.0;
            const double  dataDur1 = 20.0;
            const double  testDur1 = 10.0;
            const long    blk1 = 256;
            const long    inLen1 = round_long(dataDur1 * fs);
            const long    outLen1 = round_long(testDur1 * fs);

            std::vector<float> inSine;
            generate_steady_sine(inSine, inLen1, fc1 / fs);

            InterpPack ip1;
            MipMapFlt  mm1;
            mm1.init_sample(
                inLen1,
                InterpPack::get_len_pre(),
                InterpPack::get_len_post(),
                12,
                ResamplerFlt::_fir_mip_map_coef_arr,
                ResamplerFlt::MIP_MAP_FIR_LEN
            );
            mm1.fill_sample(&inSine[0], inLen1);

            ResamplerFlt rs1;
            rs1.set_sample(mm1);
            rs1.set_interp(ip1);
            rs1.clear_buffers();

            std::vector<float> outSine(outLen1, 0.0f);
            for (long pos = 0; pos < outLen1; pos += blk1)
            {
                const long depth = 12L << ResamplerFlt::NBR_BITS_PER_OCT;
                const long off = -10L << ResamplerFlt::NBR_BITS_PER_OCT;
                double     rat = double(pos) / double(outLen1);
                long       pitch = round_long(depth * rat) + off;
                rs1.set_pitch(pitch);

                long n = std::min(blk1, outLen1 - pos);
                rs1.interpolate_block(&outSine[pos], n);
            }
            for (auto& v : outSine) v *= 0.5f;

            // --- SAW SWEEP (10s), map level = 12 --------------------------
            const long   wave2 = 1L << 10;
            const double testDur2 = 10.0;
            const long   blk2 = 57;
            const long   inLen2 = wave2 * blk2 * 4;
            const long   outLen2 = round_long(testDur2 * fs);

            std::vector<float> inSaw;
            generate_steady_saw(inSaw, inLen2, wave2);

            InterpPack ip2;
            MipMapFlt  mm2;
            mm2.init_sample(
                inLen2,
                InterpPack::get_len_pre(),
                InterpPack::get_len_post(),
                12,
                ResamplerFlt::_fir_mip_map_coef_arr,
                ResamplerFlt::MIP_MAP_FIR_LEN
            );
            mm2.fill_sample(&inSaw[0], inLen2);

            ResamplerFlt rs2;
            rs2.set_sample(mm2);
            rs2.set_interp(ip2);
            rs2.clear_buffers();

            std::vector<float> outSaw(outLen2, 0.0f);
            for (long pos = 0; pos < outLen2; pos += blk2)
            {
                const long depth = 12L << ResamplerFlt::NBR_BITS_PER_OCT;
                const long off = -2L << ResamplerFlt::NBR_BITS_PER_OCT;
                double     rat = double(pos) / double(outLen2);
                long       pitch = round_long(depth * rat) + off;
                rs2.set_pitch(pitch);

                // periodic wrap as in main.cpp
                Int64 ppos2 = rs2.get_playback_pos();
                if ((ppos2 >> 32) > (inLen2 >> 1))
                {
                    ppos2 &= (Int64(wave2) << 32) - 1;
                    ppos2 += Int64(wave2 * 16) << 32;
                    rs2.set_playback_pos(ppos2);
                }

                long n = std::min(blk2, outLen2 - pos);
                rs2.interpolate_block(&outSaw[pos], n);
            }
            for (auto& v : outSaw) v *= 0.5f;

            // --- SINE SNAP (6s steps) with padding, map level = 12 ----------
            const double snapDur = 6.0;
            const int    steps = (2 - (-10)) + 1;      // -10..+2 octaves
            const double testDur3 = snapDur * steps;      // 78s
            const double padSec = 1.0;                  // extra pad
            const long   inLen3 = round_long((testDur3 + padSec) * fs);
            const long   outLen3 = round_long(testDur3 * fs);
            const long   blk3 = blk1;

            std::vector<float> inSnap;
            generate_steady_sine(inSnap, inLen3, fc1 / fs);

            InterpPack ip3;
            MipMapFlt  mm3;
            mm3.init_sample(
                inLen3,
                InterpPack::get_len_pre(),
                InterpPack::get_len_post(),
                12,
                ResamplerFlt::_fir_mip_map_coef_arr,
                ResamplerFlt::MIP_MAP_FIR_LEN
            );
            mm3.fill_sample(&inSnap[0], inLen3);

            ResamplerFlt rs3;
            rs3.set_sample(mm3);
            rs3.set_interp(ip3);
            rs3.clear_buffers();

            std::vector<float> outSnap(outLen3, 0.0f);
            for (long pos = 0; pos < outLen3; pos += blk3)
            {
                int    idx = int(pos / round_long(snapDur * fs));
                long   off3 = (long(-10 + idx)) << ResamplerFlt::NBR_BITS_PER_OCT;
                rs3.set_pitch(off3);

                long n = std::min(blk3, outLen3 - pos);
                rs3.interpolate_block(&outSnap[pos], n);
            }
            for (auto& v : outSnap) v *= 0.5f;

            // --- PAD & COMBINE ----------------------------------------------
            const long pad = 256;  // silence between tests
            testBuffer.clear();
            testBuffer.reserve(
                pad + outSine.size() +
                pad + outSaw.size() +
                pad + outSnap.size()
            );

            testBuffer.insert(testBuffer.end(), pad, 0.0f);
            testBuffer.insert(testBuffer.end(), outSine.begin(), outSine.end());
            testBuffer.insert(testBuffer.end(), pad, 0.0f);
            testBuffer.insert(testBuffer.end(), outSaw.begin(), outSaw.end());
            testBuffer.insert(testBuffer.end(), pad, 0.0f);
            testBuffer.insert(testBuffer.end(), outSnap.begin(), outSnap.end());
        }

    public:
        template <int P>
        void setParameter(double v)
        {
            if (P == 0)
            {
                bool newState = (v > 0.5);
                if (newState && !playToggle)
                {
                    playToggle = true;
                    playPosition = 0;
                }
                else if (!newState && playToggle)
                {
                    playToggle = false;
                }
            }
        }

        void handleHiseEvent(HiseEvent& e) {}
        SN_EMPTY_PROCESS_FRAME;

        void createParameters(ParameterDataList& data)
        {
            parameter::data p("PlayTests", { 0.0, 1.0, 1.0 });
            p.setDefaultValue(0.0);
            registerCallback<0>(p);
            data.add(std::move(p));
        }
    };

} // namespace project
